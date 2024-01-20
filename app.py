from chalice import Chalice
from chalicelib.api import TrainProgramApi
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import logging

app = Chalice(app_name='aws-lambda-deployment')
app.log.setLevel(logging.DEBUG)

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
class UserInput(BaseModel):
    type: str
    gender: str
    level: str
    frequency: int = None
    goal: str
    size: int
    weight: int
    age: int

# Need all user input plus evidences as dict
class MethodsInput(UserInput):
    evidences: dict

# Need all user input plus methods as dict
class ProgramInput(UserInput):
    methods: dict

@app.route('/provides_evidences', methods=['POST'], content_types=['application/json'])
async def provide_evidences():
    request = app.current_request
    userInput = UserInput(**request.json_body)
    trainProgramApi = TrainProgramApi(gptJsonModel, userInput.dict())
    evidences = trainProgramApi.provide_evidences()
    return evidences

@app.route('/generate_methods', methods=['POST'], content_types=['application/json'])
async def generate_methods():
    request = app.current_request
    methodsInput = MethodsInput(**request.json_body)
    trainProgramApi = TrainProgramApi(gptJsonModel, methodsInput.dict())
    evidences = trainProgramApi.provide_evidences()
    methods = trainProgramApi.generate_methods(evidences)
    return methods

@app.route('/generate_program', methods=['POST'], content_types=['application/json'])
async def generate_program(input_data: ProgramInput):
    request = app.current_request
    programInput = ProgramInput(**request.json_body)
    trainProgramApi = TrainProgramApi(gptJsonModel, programInput.dict())
    evidences = trainProgramApi.provide_evidences()
    methods = trainProgramApi.generate_methods(evidences)
    program = trainProgramApi.generate_program(methods)
    return program

# For the full workflow, only the User Input is needed
@app.route('/generate_train_program', methods=['POST'], content_types=['application/json'])
async def generate_train_program():
    request = app.current_request
    userInput = UserInput(**request.json_body)
    trainProgramApi = TrainProgramApi(gptJsonModel, userInput.dict())
    # Run the end-to-end workflow
    # TODO : write the workflow code below instead of a class method
    trainProgramApi.run_workflow()

# Ping GET endpoint for testing purposes
@app.route('/ping')
def index():
    return {'hello': 'world'}

# Response time is too long, split the workflow in multiple endpoints
@app.route('/program', methods=['POST'], content_types=['application/json'])
def create_program():
    request = app.current_request
    instance = ProgramInput(**request.json_body)
    result = TrainProgramApi(gptJsonModel, instance.dict())
    return result