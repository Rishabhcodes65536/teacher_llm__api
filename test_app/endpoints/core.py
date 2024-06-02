from urllib.request import urlopen
from bs4 import BeautifulSoup
from llama_index import Document
from llama_index import GPTSimpleVectorIndex
from llama_index import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import PromptHelper
from lang_chain_wrapper import ChatGPTLLMPredictor
from pymongo import MongoClient
import google.generativeai as genai
import json
import random 
import requests
import torch
import os
import json
import pandas as pd
from transformers import pipeline
#from unsloth import FastLanguageModel
#from transformers import AutoModelForCausalLM, AutoTokenizer

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

'''
# Initialize the text generation pipeline
text_generator = pipeline(
    "text-generation",
    model="unsloth/llama-3-8b-Instruct-bnb-4bit",
    tokenizer="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_new_tokens=1000
)'''
'''
config = {
    "hugging_face_username":"Rishabh-sucks-at-code",
    "model_config": {
        "base_model":"unsloth/llama-3-8b-Instruct-bnb-4bit", # The base model
        "finetuned_model":"llama-3-8b-Instruct-bnb-4bit-aiaustin-demo", # The finetuned model
        "max_seq_length": 2048, # The maximum sequence length
        "dtype":torch.float16, # The data type
        "load_in_4bit": True, # Load the model in 4-bit
    },
    "lora_config": {
      "r": 16, # The number of LoRA layers 8, 16, 32, 64
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], # The target modules
      "lora_alpha":16, # The alpha value for LoRA
      "lora_dropout":0, # The dropout value for LoRA
      "bias":"none", # The bias for LoRA
      "use_gradient_checkpointing":True, # Use gradient checkpointing
      "use_rslora":False, # Use RSLora
      "use_dora":False, # Use DoRa
      "loftq_config":None # The LoFTQ configuration
    },

    # "training_config": {
    #     "per_device_train_batch_size": 2, # The batch size
    #     "gradient_accumulation_steps": 4, # The gradient accumulation steps
    #     "warmup_steps": 5, # The warmup steps
    #     "max_steps":0, # The maximum steps (0 if the epochs are defined)
    #     "num_train_epochs": 10, # The number of training epochs(0 if the maximum steps are defined)
    #     "learning_rate": 2e-4, # The learning rate
    #     "fp16": torch.cuda.is_bf16_supported(), # The fp16
    #     "bf16": False, # The bf16
    #     "logging_steps": 1, # The logging steps
    #     "optim" :"adamw_8bit", # The optimizer
    #     "weight_decay" : 0.01,  # The weight decay
    #     "lr_scheduler_type": "linear", # The learning rate scheduler
    #     "seed" : 42, # The seed
    #     "output_dir" : "outputs", # The output directory
    # }
}


# Loading the model and the tokinizer for the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config.get("model_config").get("base_model"),
    max_seq_length = config.get("model_config").get("max_seq_length"),
    dtype = config.get("model_config").get("dtype"),
    load_in_4bit = config.get("model_config").get("load_in_4bit"),
)

# Setup for QLoRA/LoRA peft of the base model
model = FastLanguageModel.get_peft_model(
    model,
    r = config.get("lora_config").get("r"),
    target_modules = config.get("lora_config").get("target_modules"),
    lora_alpha = config.get("lora_config").get("lora_alpha"),
    lora_dropout = config.get("lora_config").get("lora_dropout"),
    bias = config.get("lora_config").get("bias"),
    use_gradient_checkpointing = config.get("lora_config").get("use_gradient_checkpointing"),
    random_state = 42,
    use_rslora = config.get("lora_config").get("use_rslora"),
    use_dora = config.get("lora_config").get("use_dora"),
    loftq_config = config.get("lora_config").get("loftq_config"),
)'''


#model_name = config.get("model_config").get("base_model")
#model = AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# Setup for QLoRA/LoRA peft of the base model
'''
if config.get("lora_config"):
    r = config.get("lora_config").get("r")
    target_modules = config.get("lora_config").get("target_modules")
    lora_alpha = config.get("lora_config").get("lora_alpha")
    lora_dropout = config.get("lora_config").get("lora_dropout")
    bias = config.get("lora_config").get("bias")
    use_gradient_checkpointing = config.get("lora_config").get("use_gradient_checkpointing")
    random_state = 42
    use_rslora = config.get("lora_config").get("use_rslora")
    use_dora = config.get("lora_config").get("use_dora")
    loftq_config = config.get("lora_config").get("loftq_config")

    # Modify the model according to QLoRA/LoRA setup
    model = get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        use_dora=use_dora,
        loftq_config=loftq_config,
    )

'''

# Connect to MongoDB
client = MongoClient("mongodb+srv://sanyam12sks:jP4J5DQXdcSKPwGd@cluster0.8g19w8t.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['knolly']  # Replace 'your_database' with your actual database name
collection = db['doubts']
import json

def make_api_request(content):
    url = "http://20.42.62.249:11434/api/chat"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ], "stream": False
    }

   
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response.text)
    final_response=response.json()
    res = final_response["message"]["content"]
    return res
    


def analyse_with_top(body):
    number = body.get("number")
    db_col = db['quests']

    pipeline = [
        {
            "$group": {
                "_id": "$student_id",
                "correct_answers": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$allocated_marks", "$total_marks"]},
                            1,
                            0
                        ]
                    }
                }
            }
        },
        {
            "$sort": {
                "correct_answers": -1
            }
        },
        {
            "$limit": number
        },
        {
            "$project": {
                "_id": 0,
                "student_id": "$_id"
            }
        }
    ]

    result = list(db_col.aggregate(pipeline))
    top_students = [doc["student_id"] for doc in result]
    collection = db["users"]
    student_list = []
    for student_id in top_students:
        student = collection.find_one({'_id': student_id})
        student_list.append(student['name'].split()[0])
    return {"status": 1, "student_rank_with_order": student_list}

def coaching_with_llm(body):
    try:
        response_dict = coaching_with_llm_2(body)
    except Exception as e:
        print(f"Exception: {e}")
        response_dict = coaching_with_llm_2(body)
    return response_dict

def coaching_with_llm_2(body):
    student_question = body.get("question")
    answer = body.get("answer")
    marks = body.get("marks")
    final_answer = body.get("final_answer")

    query_1 = f"Solve this question: {student_question}. Store the response in a JSON format where you have {{'solution_steps': solution_string, 'final_answer':final_answer_string}} Make sure the solution_steps are brief. solution_steps and final_answer are strings.  Return a valid json and nothing else."
    content = query_1
    response = make_api_request(content)
    print("first response", response)
    try:
        json_response_1 = extract_json(response)
        json_data_1 = json.loads(json_response_1)
    except:
        response = make_api_request(content)
        print("second response",response)
        json_response_1 = extract_json(response)
        json_data_1 = json.loads(json_response_1)

    model_solution = json_data_1["solution_steps"]
    model_final = json_data_1["final_answer"]

    final_query = (
        f"Here is a Question: {student_question}\n"
        f"Here is the Correct Solution: {model_solution}\n"
        f"Maximum marks is {marks}\n"
        f"Student have given this solution: {answer}\n"
        f"Check if the student solution is correct and award marks based on correctness of student answer then respond a JSON like "
        f"{{'marks_awarded_to_student': , 'feedback':''}}. Do not use any apostrophe in feedback.Answer only in JSON format so it is easier to extract. Do not return nothing apart from Json."
    )
    content = final_query

    try:
        response = make_api_request(content)
        print("third response", response)
        json_response = extract_json(response)
        data = json.loads(json_response)
    except Exception as e:
        print(f"Exception: {e}")
        response = make_api_request(content)
        json_response = extract_json(response)
        data = json.loads(json_response)
        print("fourth response", response)

    if data:
        marks_awarded_to_student = data.get('marks_awarded_to_student', 0)
    else:
        marks_awarded_to_student = 0

    try:
        if int(model_final) == int(final_answer):
            data["marks_awarded_to_student"] = marks
    except:
        print("Final answer not matching exactly")

    if str(model_final) == str(final_answer) or marks_awarded_to_student == marks:
        data["marks_awarded_to_student"] = marks
    elif data["marks_awarded_to_student"] != marks:
        comparison_query = (
            f"First answer: {final_answer} and second answer: {model_final}. Compare and check whether the two answers match. "
            f"If they do, then return a JSON {{'total_marks': {marks}}}. Otherwise, return total_marks as {{'total_marks': 0}} \n"
            f"Remember only return a valid JSON"
        )
        content= comparison_query
        try:
            response = make_api_request(content)
            print("last query", response)
            json_response = extract_json(response)
            comparison_data = json.loads(json_response)
            data["marks_awarded_to_student"] = comparison_data.get("total_marks", marks_awarded_to_student)
        except:
            print("Query failed")
    return {"status": 1, "response":data,"question":student_question, "student_answer":answer}

def store_interaction(question, response):
    interaction = {"question": question, "response": response}
    collection.insert_one(interaction)

def convert_timestamp(doc):
    doc["timestamp"] = Int64(doc["timestamp"]["$date"]["$numberLong"])
    return doc
def analyse_with_llm_chat(body):
    student_question = body.get("question")
    student_id = body.get("student_id")
    student_name = body.get("student_name")
    response_type = body.get("response_type")

    if response_type == "follow_up":
        limit = 1
        pipeline = [
            {"$match": {"student_id": student_id}},
            {"$sort": {"timestamp": -1}},
            {"$limit": limit}
        ]
        past_interactions = list(collection.aggregate(pipeline))
        context = ""
        for interaction in past_interactions:
            context += interaction["question"] + "\n" + interaction["answer"] + "\n"
    else:
        context = ""

    instructions = """
    Task is to be a maths buddy who is helping students practising for GCSE exam. Here are some contextual data. Some revision tips for GCSE are:
    Tip 1: The best way to revise GCSE Maths is to DO lots of Maths.
    Tip 2: Revise lots of different topics in rotation.
    Tip 3: Try some exam questions, fill in the gaps, then go back and try again.
    Tip 4: Understand the mark scheme
    Tip 5: Gradually reduce reliance on notes and formula sheets.
    Tip 6: Explain what you're doing.
    Tip 7: Don't throw away easy marks.
    Tip 8: Build up to exam conditions.    
    Tip 9: Keep an eye on the clock.
    Tip 10: You don't have to do the questions in order.
    Tip 11: Look beyond your exam board.
    Tip 12: Don't overdo it!
    Here methods how to prepare about GCSE but inefficient techniques such as rereading, highlighting, and summarizing, based on research findings. Instead, it recommends active recall and spaced repetition as more effective strategies. The piece delves into various study methods like closed-book exercises, Anki flashcards, and the Feynman Technique. It also offers tips for effective note-taking, including the Cornell Method and mind maps.
    GCSE maths syllabus:
    The GCSE maths test comprises six subject areas: Number, Algebra, Ratio, Proportion and Rates of Change, Geometry and Measures, Probability, and Statistics. It offers two tiers: Foundation (grades 1–5) and Higher (grades 4–9). Each tier requires three question papers to be taken in the same series.
    The topics covered in the syllabus include:
    Number: Covers operations with integers, decimals, fractions, prime numbers, powers, standard form, etc.
    Algebra: Focuses on algebraic notation, manipulation, graphs, solving equations and inequalities, sequences, etc.
    Ratio, Proportion and Rates of Change: Addresses the manipulation of ratios, proportions, percentages, and solving problems involving direct and inverse proportion.
    Geometry and Measures: Includes properties of geometric figures, mensuration, vectors, etc.
    Probability: Covers concepts of probability, frequency analysis, theoretical probability, and constructing possibility spaces.
    Statistics: Focuses on interpreting and constructing various types of data representations, applying statistics to describe populations, and analyzing scatter graphs.
    """

    final_query = f"You are a maths teacher who is helping out {student_name} in his maths journey. Here is the sentence said by {student_name}: {student_question}. Here is the past conversation Context:\n{context}. Your name is Knolly. Answer {student_name}'s query also give him an insight about his question which can be an application of his maths query. Greet only sometimes."

    content = instructions + "\n\n" + final_query

    try:
        response = make_api_request(content)
    except:
        response = make_api_request(content)

    return {"status": 1, "response": response}

def analyse_with_llm(body):
    student_question = body.get("question")
    past_question = body.get("past_question")
    topic, subtopic = select_random_topic(student_question)
    instructions = """
    You are a maths teacher. You have to recommend one question based on the topic asked by the student. Assign marks to it as well. Try to generate a new topic of the user question.
    """

    final_query = (
        f"Here is the topic and subtopic asked by the student: {topic} and {subtopic}. "
        f"Assign marks to it as well. Try to generate a new topic of the user question. Make sure the marks are random and always less than 6 "
        f"Past Question was {past_question}\n"
        "Answer the question in the JSON format with {'question': 'your_generated_question_here', 'marks': your_assigned_marks}. "
        "Ensure that the response is a valid JSON object and do not use any apostrophe in the question."
    )

    content = instructions + "\n\n" + final_query

    
    response = make_api_request(content)
    # Extract JSON part from the response
    print(response)
    try:
        json_response = extract_json(response)
        if json_response:
            data = json.loads(json_response)
    except:
        json_response = extract_json(response)
        if json_response:
            data = json.loads(json_response)
    print(f"Data: {data}")  
    
    return {"status": 1, "response": data}




def health():
    return "OK"



def doc_builder_coaching():
    
    #doc = f" The scraped data for the URL {url} are" + url_data
    instructions = ("Task is to rate student based on their answers. You will get a question, total marks and students solution.\n"
               "Instructions:\n"
               "• Only check the final answer given by the student.\n"
               "• Give a feedback to the student \n"
               "• Allot marks to the student. \n"
               "• The response should be in json format with {'marks_awarded_to_student': '', 'feedback': }.\n")


    doc = f"Here are some sample reference  {instructions}"
    doc = Document(doc)

    return doc


def doc_builder_question_compare(answer_1, answer_2):
    
    instructions = ("You are a maths intructor who is good at evaluating final answers. You would be given two answer and you have to tell if both of them are same or not with numbers. Your response should be {'total_marks': ''} ")
    doc = f"Here are some sample reference  {instructions}"
    doc = Document(doc)

    return doc

def index_builder_question_compare(model, answer_1, answer_2):
    docs = []
    model = "gpt-3.5-turbo"
    role = "maths teacher"
    docs.append(doc_builder_question_compare(answer_1, answer_2))
    content = "You are a helpful " + role + "assistant"
    llm_predictor = ChatGPTLLMPredictor([{"role": "system", "content": content}], model)
    model_name = "sentence-transformers/stsb-mpnet-base-v2"
    hf_embedd = HuggingFaceEmbeddings(model_name=model_name)
    embed_model = LangchainEmbedding(hf_embedd)
    max_input_size = 8000
    num_output = 512
    max_chunk_overlap = 200
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    index = GPTSimpleVectorIndex(docs, embed_model=embed_model, llm_predictor=llm_predictor,
                                    prompt_helper=prompt_helper)
    return index

def doc_builder():
    
    #doc = f" The scraped data for the URL {url} are" + url_data
    instructions = ("Task is to generate similar questions and assign marks to each question as per the difficulty.\n"
               "Instructions:\n"
               "• You need to change in the numbers with the question\n"
               "• Try to rephrase the questions\n"
               "• Make it a new question\n"
               "• The response should be in json format with {'question': '', 'marks': }.")

    similar_question = ("Here are some sample questions\n"
                    "1. Here is a list of numbers 2 4 4 7 8\n Work out the range of these numbers. Marks= 1\n"
                    "2. Work out 120 – 89 Marks= 1\n"
                    "3. Simplify 3×a×4 Marks= 1\n"
                    "4. There are 3 litres of oil in a can. Jermaine uses 700 millilitres of the oil. Work out the amount of oil left in the can. Give your answer in millilitres. Marks= 3 \n"
                    "5. Ebony makes some bracelets to sell. The materials to make all the bracelets cost £190, correct to the nearest £5Ebony sells all the bracelets for a total of £875, correct to the nearest £5The total time taken to make and sell all these bracelets was 72 hours, correct to thenearest hour.Ebony uses this method to calculate her hourly rate of payHourly rate of paytotal selling price total cost of materials total time taken The minimum hourly rate of pay for someone of Ebony’s age is £8.20 By considering bounds, determine if Ebony’s hourly rate of pay was definitely morethan £8.20 You must show all your working. Marks = 4 \n"
                    "6. Expand and simplify 5( p + 3) – 2(1 – 2p) Marks 2 \n"
                    "7. Lamp A flashes every 20 seconds.Lamp B flashes every 45 seconds.Lamp C flashes every 120 seconds. The three lamps start flashing at the same time. How many times in one hour will the three lamps flash at the same time? Marks 3 \n")

    doc = f"Here are some sample questions for your reference. {similar_question}"
    doc = Document(doc)

    return doc

def doc_builder_chat():
    

    instructions = """
    Task is to be a maths buddy who is helping students practising for GCSE exam. Here are some contextual data. Some revision tips for GCSE are:
    Tip 1: The best way to revise GCSE Maths is to DO lots of Maths.
    Tip 2: Revise lots of different topics in rotation.
    Tip 3: Try some exam questions, fill in the gaps, then go back and try again.
    Tip 4: Understand the mark scheme
    Tip 5: Gradually reduce reliance on notes and formula sheets.
    Tip 6: Explain what you're doing.
    Tip 7: Don't throw away easy marks.
    Tip 8: Build up to exam conditions.    
    Tip 9: Keep an eye on the clock.
    Tip 10: You don't have to do the questions in order.
    Tip 11: Look beyond your exam board.
    Tip 12: Don't overdo it!
    Here methods how to prepare about GCSE but inefficient techniques such as rereading, highlighting, and summarizing, based on research findings. Instead, it recommends active recall and spaced repetition as more effective strategies. The piece delves into various study methods like closed-book exercises, Anki flashcards, and the Feynman Technique. It also offers tips for effective note-taking, including the Cornell Method and mind maps.
    GCSE maths syllabus:
    The GCSE maths test comprises six subject areas: Number, Algebra, Ratio, Proportion and Rates of Change, Geometry and Measures, Probability, and Statistics. It offers two tiers: Foundation (grades 1–5) and Higher (grades 4–9). Each tier requires three question papers to be taken in the same series.
    The topics covered in the syllabus include:
    Number: Covers operations with integers, decimals, fractions, prime numbers, powers, standard form, etc.
    Algebra: Focuses on algebraic notation, manipulation, graphs, solving equations and inequalities, sequences, etc.
    Ratio, Proportion and Rates of Change: Addresses the manipulation of ratios, proportions, percentages, and solving problems involving direct and inverse proportion.
    Geometry and Measures: Includes properties of geometric figures, mensuration, vectors, etc.
    Probability: Covers concepts of probability, frequency analysis, theoretical probability, and constructing possibility spaces.
    Statistics: Focuses on interpreting and constructing various types of data representations, applying statistics to describe populations, and analyzing scatter graphs.
    """

    doc = f" {instructions}"
    doc = Document(doc)

    return doc

def index_builder_coaching(model):

    docs = []
    model = "gpt-3.5-turbo"
    role = "maths teacher"
    docs.append(doc_builder_coaching())
    content = "You are a helpful " + role + "assistant"
    llm_predictor = ChatGPTLLMPredictor([{"role": "system", "content": content}], model)
    model_name = "sentence-transformers/stsb-mpnet-base-v2"
    hf_embedd = HuggingFaceEmbeddings(model_name=model_name)

    # load in HF embedding model from langchain
    embed_model = LangchainEmbedding(hf_embedd)
    max_input_size = 8000
    # set number of output tokens
    num_output = 512
    # set maximum chunk overlap
    max_chunk_overlap = 200

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    index = GPTSimpleVectorIndex(docs, embed_model=embed_model, llm_predictor=llm_predictor,
                                    prompt_helper=prompt_helper)

    return index


def index_builder(model):

    docs = []
    model = "gpt-3.5-turbo"
    role = "maths teacher"
    docs.append(doc_builder())
    content = "You are a helpful " + role + "assistant"
    llm_predictor = ChatGPTLLMPredictor([{"role": "system", "content": content}], model)
    model_name = "sentence-transformers/stsb-mpnet-base-v2"
    hf_embedd = HuggingFaceEmbeddings(model_name=model_name)

    # load in HF embedding model from langchain
    embed_model = LangchainEmbedding(hf_embedd)
    max_input_size = 8000
    # set number of output tokens
    num_output = 512
    # set maximum chunk overlap
    max_chunk_overlap = 200

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    index = GPTSimpleVectorIndex(docs, embed_model=embed_model, llm_predictor=llm_predictor,
                                    prompt_helper=prompt_helper)

    return index

def index_builder_chat(model):

    docs = []
    model = "gpt-3.5-turbo"
    role = "maths teacher"
    docs.append(doc_builder_chat())
    content = "You are a helpful " + role + "assistant"
    llm_predictor = ChatGPTLLMPredictor([{"role": "system", "content": content}], model)
    model_name = "sentence-transformers/stsb-mpnet-base-v2"
    hf_embedd = HuggingFaceEmbeddings(model_name=model_name)

    # load in HF embedding model from langchain
    embed_model = LangchainEmbedding(hf_embedd)
    max_input_size = 8000
    # set number of output tokens
    num_output = 512
    # set maximum chunk overlap
    max_chunk_overlap = 200

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    index = GPTSimpleVectorIndex(docs, embed_model=embed_model, llm_predictor=llm_predictor,
                                    prompt_helper=prompt_helper)

    return index

def select_random_topic(subject):

    exam_topics = {
        "Algebra problem": {
            "Algebraic Manipulation": [
                "Changing the subject advanced",
                "Collecting like terms",
                "Completing the square",
                "Dividing terms",
                "Expanding brackets",
                "Expanding two brackets",
                "Expanding three brackets",
                "Expressions – forming",
                "Indices",
                "Multiplying terms",
                "Notation",
                "Substitution"
            ],
            "Algebraic Fractions": [
                "Addition",
                "Division",
                "Multiplication",
                "Simplifying"
            ],
            "Equation of a Circle": [
                "Center-radius form equation",
                "Finding radius of the circle"
            ],
            "Simple maths": [
                "Simple Factorization",
                "Rearranging equations",
                "Linear inequalities",
                "Quadratic inequalities",
                "quadratic inequality by factorising",
            ],
            "Sequences": [
                "Arithmetic sequences",
                "Geometric sequences",
                "Special sequences",
                "Quadratic sequences",
                "Fibonacci Sequence",

            ],
            "Functions": ["Composite functions", "Inverse functions", ]

        },
        "Statistics problem": {
            "Averages": [
                "Mean",
                "Median",
                "Mode",
                "Range",
                "Quartiles"
            ],
            "Frequency Tables": [
                "Mean",
                "Median",
                "Mode",
                "Range"
            ],
            "Estimation": [
                "Mean",
                "Median"
            ],
        },
        "Number Theory problem": {
            "Multiplication": [
                "Grid method",
                "Column method",
                "End number",
                "By 10, 100 etc",
                "Powers of 10",
                "Decimals",
                "Times tables"
            ],
            "Negatives": [
                "Addition/subtraction",
                "Multiplication",
                "Division",
                "Ordering",
                "Real-life applications"
            ],
            "Basic Operations": [
                "Best buys",
                "BODMAS (order of operations)"
            ],
            "Factors and Multiples": [
                "Common multiples/LCM",
                "Common factors/HCF",
                "Multiples",
                "Ordering numbers"
            ],
            "Number Properties": [
                "Cube numbers",
                "Cubing a number",
                "Cube root",
                "Currency",
                "Estimation",
                "Place value",
                "Odd and even numbers",
                "Prime numbers",
                "Square numbers",
                "Squaring a number",
                "Square root",
                "Triangular numbers",
                "Rational/irrational numbers"
            ],
            "Product of Primes": [
                "Product of primes",
                "LCM/HCF"
            ],
        },
        "Probability problem": {
            "Basic Probability": [
                "Basic probability",
                "Sample space",
                "Relative frequency"
            ],
            "Rules of Probability": [
                "OR rule",
                "Conditional probability",
                "Independent events",
                "Not happening",
                "Union",
                "Intersection",
                "Complement",
                "Dependent Events",
                "Mutually exclusive events",
                "Exhaustive events",
            ],
            "Probability Techniques": [
                "Tree diagrams",
                "Listing outcomes"
            ]
        },
        "Differentiation problem": {
            "Basic Differentiation": [
                "Finding derivatives of basic functions",
                "Using the power rule for differentiation",
                "Using the constant multiple rule for differentiation"
            ],
            "Differentiation of Composite Functions": [
                "Applying the chain rule",
                "Finding derivatives of composite functions"
            ],
            "Implicit Differentiation": [
                "Applying implicit differentiation to find derivatives"
            ],
            "Differentiation of Exponential and Logarithmic Functions": [
                "Finding derivatives of exponential functions",
                "Finding derivatives of logarithmic functions",
                "Using the rules of differentiation for exponentials and logarithms"
            ],
            "Differentiation of Trigonometric Functions": [
                "Finding derivatives of trigonometric functions",
                "Applying the chain rule to trigonometric functions"
            ],
            "Applications of Differentiation": [
                "Finding equations of tangents and normals",
                "Finding maximum and minimum points",
                "Solving optimization problems"
            ],
            "Related Rates": [
                "Understanding related rates problems",
                "Applying differentiation to solve related rates problems"
            ]
        },
        "Integration problem": {
            "Basic Integration": [
                "Finding indefinite integrals of basic functions",
                "Using the power rule for integration",
                "Using the constant multiple rule for integration"
            ],
            "Integration by Substitution": [
                "Identifying appropriate substitutions",
                "Applying the substitution method to solve integrals"
            ],
            "Integration by Parts": [
                "Understanding when to use integration by parts",
                "Applying the integration by parts formula",
                "Repeated integration by parts"
            ],
            "Integration of Trigonometric Functions": [
                "Integrating trigonometric functions",
                "Using trigonometric identities to simplify integrals"
            ],
            "Integration of Exponential and Logarithmic Functions": [
                "Integrating exponential functions",
                "Integrating logarithmic functions",
                "Using properties of logarithms to simplify integrals"
            ],
            "Applications of Integration": [
                "Finding areas under curves",
                "Finding the area between curves",
                "Finding the average value of a function over an interval"
            ],
            "Definite Integrals": [
                "Understanding the difference between definite and indefinite integrals",
                "Evaluating definite integrals using the fundamental theorem of calculus",
                "Interpreting definite integrals as areas"
            ]
        },
        "Linear Algebra problem": {
        "Matrix Operations": [
            "Addition",
            "Subtraction",
            "Scalar multiplication",
            "Matrix multiplication",
            "Determinants"
        ],
        "Systems of Linear Equations": [
            "Solving systems of linear equations",
            "Gaussian elimination",
            "Matrix methods"
        ],
        "Vector Operations": [
            "Addition",
            "Subtraction",
            "Scalar multiplication",
            "Vector magnitude",
            "Dot product",
            "Cross product"
        ]
    },
    "Geometry problem": {
        "Angle Properties": [
            "Angle sum of polygons",
            "Interior and exterior angles of polygons",
            "Properties of parallel lines",
            "Properties of perpendicular lines"
        ],
        "Coordinate Geometry": [
            "Distance formula",
            "Midpoint formula",
            "Equation of a line",
            "Gradient of a line"
        ],
        "Geometric Transformations": [
            "Translation",
            "Reflection",
            "Rotation",
            "Enlargement"
        ],
        "Pythagoras' Theorem": [
            "Using Pythagoras' theorem to find missing lengths",
            "Finding the length of the hypotenuse",
            "Finding the length of a shorter side"
        ]
    },
    "Trigonometry problem": {
        "Trigonometric Functions": [
            "Sine function",
            "Cosine function",
            "Tangent function",
            "Cosecant function",
            "Secant function",
            "Cotangent function"
        ],
        "Solving Trigonometric Equations": [
            "Using trigonometric identities",
            "Using inverse trigonometric functions",
            "Solving equations involving multiple trigonometric functions"
        ],
        "Applications of Trigonometry": [
            "Finding distances and heights",
            "Modeling periodic phenomena",
            "Analyzing alternating currents",
            "Navigational problems"
        ]
        }
    }
    if subject in exam_topics:
        topic = random.choice(list(exam_topics[subject].keys()))
        subtopic = random.choice(exam_topics[subject][topic])
        return topic, subtopic
    else:
        return None, None


def query_builder(index, question, past_question):

    subject = question
    try:
        topic, subtopic = select_random_topic(subject)
        final_query = f"You are a maths teacher. You have to recommend questions based on the topic asked by the student.\n" \
              f"Here is the topic and subtopic asked by the student: {topic} {subtopic}  Assign marks to it as well. Try to generate a new topic of the user question. Past Question was {past_question}\n" \
              "Answer the question in this format with {{'question': '', 'marks': ''}} Do not use any apostrophe in the question" 
        response = index.query(final_query, mode="embedding")
        answer = response.response
        original_string = answer
        start_index = original_string.find('{')
        end_index = original_string.find('}')
        answer = original_string[start_index:end_index+1]
        json_string = answer.replace("'", "\"")
        print(json_string)
        data = json.loads(json_string)
    
    except:
        topic, subtopic = select_random_topic(subject)
        final_query = f"You are a maths teacher. You have to recommend questions based on the topic asked by the student.\n" \
              f"Here is the topic and subtopic asked by the student: {topic} {subtopic}  Assign marks to it as well. Try to generate a new topic of the user question. Past Question was {past_question}\n" \
              "Answer the question in this format with {{'question': '', 'marks': ''}} Do not use any apostrophe in the question" 
        response = index.query(final_query, mode="embedding")
        answer = response.response
        original_string = answer
        start_index = original_string.find('{')
        end_index = original_string.find('}')
        answer = original_string[start_index:end_index+1]
        json_string = answer.replace("'", "\"")
        print(json_string)
        data = json.loads(json_string)
    
    return data

def query_builder_old(index, question, past_question):

    print(question, past_question)
    subject = question
    topic, subtopic = select_random_topic(subject)
    final_query = f"You are a maths teacher. You have to recommend questions based on the topic asked by the student.\n" \
              f"Here is the topic and subtopic asked by the student: {topic} {subtopic}  Assign marks to it as well. Try to generate a new topic of the user question. Past Question was {past_question}\n" \
              "Answer the question in this format with {{'question': '', 'marks': ''}} Do not use any apostrophe in the question" 
    print(final_query)
    response = index.query(final_query, mode="embedding")
    #print(final_query)
    #print(past_question)
    answer = response.response
    original_string = answer
    start_index = original_string.find('{')
    end_index = original_string.find('}')
    answer = original_string[start_index:end_index+1]
    # print(answer)
    try:
        json_string = answer.replace("'", "\"")
        data = json.loads(json_string)
    except:
        response = index.query(final_query, mode="embedding")
        answer = response.response
        json_string = answer.replace("'", "\"")
        data = json.loads(json_string)
    print(data, past_question)
    return data

'''
def doc_builder_multi():
    
    instructions = ("Task is to generate a question as per student's topic and generate solution in step format.\n"
               "Instructions:\n"
               "• Check the steps followed by the student\n"
               "• Give a feedback to the student \n"
               "• Allot marks to the student. \n"
               "• The response should be in json format with {'question': '', 'steps': [{'order':1, 'solution_step': ''}, {'order':2, 'solution_step': ''}] }.\n")


    doc = f"Here are some sample reference  {instructions}"
    doc = Document(doc)

    return doc
'''

def index_builder_multi():
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

    text_generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=1000,
	temperature=0.9,
        top_k=3,
        top_p=0.97
    )
    return text_generator

'''
def query_builder_multi(index, topic, past_question):
    final_query = (
        f"You are a maths teacher. You have to recommend questions along with the answer based on the topic asked by the student. Here is the past question: {past_question}. Make sure you generate a new question.\n"
        f"Here is the topic and subtopic provided by the student: {topic}. Generate the question, and steps of answer. You can have multiple steps, final answer should be in one step only. Assign marks to the question as well. The steps should be simple to order such that formula step is before substitution step.\n"
        "Answer the question in JSON format with [{'question': '', 'marks': '', 'steps':[{'order': 1, 'solution_step':''}, {'order': 2, 'solution_step':''}]}]. The value of marks is the number of steps. Do not use any apostrophes in the question. No need to add slashes or line changes. Your response must consist a JSON no additional information"
    )

    sequences = index(final_query)
    gen_text = sequences[0]["generated_text"]
    print("Generated Text:", gen_text)
    
    try:
        json_start = gen_text.find("[")
        json_end = gen_text.rfind("]") + 1
        json_string = gen_text[json_start:json_end]
        print("JSON Start Index:", json_start)
        print("JSON End Index:", json_end)
        print("Extracted JSON String:", json_string)
        json_data = json.loads(json_string)
    except (ValueError, KeyError) as e:
        print(f"Error parsing JSON: {e}")
        json_data = {"error": "Failed to generate a valid JSON response"}

    return json_data
'''
def extract_json(response):
    # Find the starting index of JSON
    response = response.replace('\n', '')
    start_index = response.find("{")
    # Find the ending index of JSON
    end_index = response.rfind("}") + 1
    # Extract the JSON substring
    json_part = response[start_index:end_index]
    return json_part


def analyse_with_multi(body):
    student_question = body.get("question")
    if not student_question:
        # Handle the case where "question" key is missing in the body
        return {"status": 0, "error": "Missing 'question' key in the request body"}
    print(student_question)
    topic = body.get("question")
    subject = topic+ " problem"
    topic, subtopic = select_random_topic(subject)
    content = f"You are a maths teacher. You have to recommend questions along with the answer based on the topic asked by the student. Make sure you generate new question\n"
    content += f"Here is the topic and subtopic by the student: {topic} and {subtopic} . Generate the question, and steps of answer. You can have multiple steps, final answer should be in one step only. Assign marks to the question as well. The steps should be simple to order such that formula step is before substitution step.\n"
    content += "Answer the question in JSON format with [{'question': '', 'marks': '', 'steps':[{'order': 1, 'solution_step':''}, {'order':2, 'solution_step'}, {}, {}]}]. Value of marks is number of steps. Do not use any apostrophe in the question. No need to add slashes or line changes"

    print(content)
    try:
        response = make_api_request(content)
        # Extract JSON part from the response
        json_response = extract_json(response)
        data = json.loads(json_response)
    except:
        response = make_api_request(content)  
        json_response = extract_json(response)
        data = json.loads(json_response)
        
    print(response)
    return {"status": 1, "response": data}
