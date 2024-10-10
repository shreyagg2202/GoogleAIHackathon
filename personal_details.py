# personal_details.py

from pydantic import BaseModel
from typing import List

# Define PersonalDetails classes
from pydantic import BaseModel, Field
from typing import Optional

# Base class for common personal details
class BasePersonalDetails(BaseModel):
    name: Optional[str] = Field(default="", description="This is the name of the user.")
    age: Optional[str] = Field(default="", description="This is the age of the user.")
    date_of_birth: Optional[str] = Field(default="", description="This is the date of birth of the user.")
    address: Optional[str] = Field(default="", description="This is the address of the user.")
    phone_number: Optional[str] = Field(default="", description="This is the phone number of the user.")
    email_address: Optional[str] = Field(default="", description="This is the email address of the user.")

# Class for vehicle insurance details
class VehiclePersonalDetails(BasePersonalDetails):
    vehicle_age: Optional[str] = Field(default="", description="This is the age of the vehicle.")
    vehicle_details: Optional[str] = Field(default="", description="This provides details about the vehicle (make, model, etc.).")
    previous_accidents: Optional[str] = Field(default="", description="Any previous accidents involving the vehicle.")

# Class for health insurance details
class HealthPersonalDetails(BasePersonalDetails):
    allergies: Optional[str] = Field(default="", description="This indicates any known allergies of the user.")
    current_medications: Optional[str] = Field(default="", description="Details of the current medications the user is taking.")
    occupation: Optional[str] = Field(default="", description="The occupation of the user.")
    income: Optional[str] = Field(default="", description="The annual income of the user.")

# Class for life insurance details
class LifePersonalDetails(BasePersonalDetails):
    height: Optional[str] = Field(default="", description="The height of the user.")
    weight: Optional[str] = Field(default="", description="The weight of the user.")
    chronic_illnesses: Optional[str] = Field(default="", description="Details of any chronic illnesses the user has.")
    occupation: Optional[str] = Field(default="", description="The occupation of the user.")
    income: Optional[str] = Field(default="", description="The annual income of the user.")
    smoker_status: Optional[str] = Field(default="", description="The smoking status of the user (smoker or non-smoker).")

# Functions related to personal details
def check_what_is_empty(user_personal_details) -> List[str]:
    ask_for = []
    # Check if fields are empty
    for field, value in user_personal_details.dict().items():
        if value in [None, "", 0]:
            ask_for.append(field)
    return ask_for

def add_non_empty_details(current_details, new_details):
    if new_details is not None:
        non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
        updated_details = current_details.copy(update=non_empty_details)
        return updated_details
    return current_details
