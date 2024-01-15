from chalice import Chalice
from workflow import Workflow
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

gptJsonModel = ChatOpenAI(
    #models : https://platform.openai.com/docs/models/gpt-3-5
    # trying GPT-4 turbo preview
    model="gpt-4-1106-preview",
    #model="gpt-3.5-turbo-1106",
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)

# Define a Pydantic model for the JSON data input
class ProgramInput(BaseModel):
    type: str
    gender: str
    level: str
    frequency: int = None
    goal: str
    size: int
    weight: int
    age: int

app = Chalice(app_name='aws-lambda-deployment')

@app.route('/ping')
def index():
    return {'hello': 'world'}

@app.route('/api/program', methods=['POST'], content_types=['application/json'])
async def create_program(input_data: ProgramInput):
    result = Workflow(gptJsonModel, input_data.dict())
    return result