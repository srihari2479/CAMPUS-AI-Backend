from fastapi import APIRouter, HTTPException, status
from app.services.supabase_client import supabase_client

router = APIRouter(prefix="/api/graph", tags=["Syllabus Knowledge Graph"])

@router.get("/department/{dept_code}/courses")
async def get_department_courses(dept_code: str):
    """
    Returns all courses offered by a specific department, filtered by code (e.g., 'CSE').
    """
    if not supabase_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Supabase client is not configured."
        )

    try:
        # Fetch department with nested courses
        response = supabase_client.table("departments") \
            .select("name, code, courses(id, name, code, semester)") \
            .eq("code", dept_code.upper()) \
            .execute()
            
        if not response.data:
            raise HTTPException(
                status_code=404, 
                detail=f"Department with code '{dept_code}' not found."
            )
            
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching department courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/course/{course_code}/structure")
async def get_course_structure(course_code: str):
    """
    Returns the complete hierarchy of a course (Course -> Units -> Topics)
    representing the structured syllabus knowledge graph.
    """
    if not supabase_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Supabase client is not configured."
        )

    try:
        # Use PostgREST relational selectors to retrieve complete hierarchy
        response = supabase_client.table("courses") \
            .select("id, name, code, semester, units(id, unit_number, title, topics(id, title, description))") \
            .eq("code", course_code.upper()) \
            .execute()
            
        if not response.data:
            raise HTTPException(
                status_code=404, 
                detail=f"Course with code '{course_code}' not found."
            )
            
        # Sort units in Python just in case to guarantee ascending order
        course_data = response.data[0]
        if "units" in course_data:
            course_data["units"] = sorted(
                course_data["units"], 
                key=lambda u: u.get("unit_number", 0)
            )
            
        return course_data
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching course structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))
