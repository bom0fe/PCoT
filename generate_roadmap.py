from openai import OpenAI
import pandas as pd
import numpy as np
import json
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import shutil
import re

from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.http import MediaFileUpload

# 스프레드시트 접근을 위한 서비스 계정 키 파일 경로
SERVICE_ACCOUNT_FILE = 'your path'

# 접근 권한 설정
scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# 자격 증명 객체 생성
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)

# Google 스프레드시트 API 클라이언트 생성
client = gspread.authorize(credentials)

# 스프레드시트 ID 또는 URL
SPREADSHEET_ID = 'your id'
SHEET_NAME = 'log'  # sheet 이름

# 스프레드시트 열기
spreadsheet = client.open_by_key(SPREADSHEET_ID)
sheet = spreadsheet.worksheet(SHEET_NAME)

# 스프레드시트 데이터 가져오기
data = sheet.get_all_records()

# 필요한 데이터 추출 및 DataFrame에 저장
log_data = [{'id': row['id'], 'userData': str(row['userData'])} for row in data]

# DataFrame 생성
df = pd.DataFrame(log_data)

# userData 열에서 개행 문자 제거 및 key-value 형식 유지
def clean_user_data(json_string):
    parsed_json = json.loads(json_string)
    cleaned_json_string = json.dumps(parsed_json, ensure_ascii=False)
    return cleaned_json_string

# DataFrame에 새로운 열 추가하여 정리된 문자열 저장
df['cleaned_userData'] = df['userData'].apply(clean_user_data)

userData = df["cleaned_userData"].values[-1]
#print(userData)

# Database
project_info = {
"Visual Object Tracking Using Plenoptic Image Sequences":"",
"Anti Drone AI Robot using object detection":"",
"AI Mashroom Classificator App":"",
"딥러닝, 쓰레기 분류 모델 만들기":"",
"시각화, 뉴욕시 에어비엔비 태블로 분석":"",
"SageMaker를 이용한 디오비스튜디오 Face Swap Model Train Pipeline 구상 및 구축":"",
"dob On-Premise Server Monitoring 구축":"",
"Private Subnet CD 환경 구축":"",
"Hand BoneAge(MediAI) 모델 개발":"",
"안내 로봇 인터렉션 데모 모델 개발":"",
"영상 모자이크 처리":"",
"X-Ray Baggage Scanner 자동 검출 솔루션":"",
"재활용품 분류를 위한 Semantic Segmentation":"",
"마스크 착용 상태 분류":"",
"합성곱 신경망 기반의 이산화탄소 지중격리 운영조건 최적설계":""
} # update 필요

client = OpenAI(api_key='your api key')
model = "gpt-4o"

# Embedding
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(v1, v2):
    # 두 벡터의 내적(dot product)
    dot_product = np.dot(v1, v2)

    # 각 벡터의 노름(norm)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 코사인 유사도 계산
    similarity = dot_product / (norm_v1 * norm_v2)

    return similarity

for description in project_info.keys():

    project_info[description] = get_embedding(description)

query_prompt="""
You are an helpful assistant who organizes the desired company, desired job, and previous projects. Please produce the results exactly the same as the form below.
\\
Form Example:
희망 기업은 {hopeCorporation}이고, 희망 직무는 {jobObjectives}이다. 관심있는 기업이 이전에 진행하는 프로젝트는 {previousProject}이다.\\
    
"""
query = client.chat.completions.create(
    model = model,
    messages = [
        {"role" : "system", "content" : query_prompt},
        {"role" : "user", "content" : userData}
    ]
).choices[0].message.content

query_embedding = get_embedding(query)

max_similarity = -1
most_similar_key = None

for key, vector in project_info.items():
    similarity = cosine_similarity(query_embedding, vector)
    if similarity > max_similarity:
      max_similarity = similarity
      most_similar_key = key

# 가장 유사한 주제 생성
most_similar_content = most_similar_key


USER_INFO_SUMMARIZER = """
Summarize the user-provided information in two English sentences while maintaining the existing semantic relationships and including all the information. Highlight important keywords using bold formatting.
"""

SUMMARIZED_USER_info = client.chat.completions.create(
    model = model,
    messages = [
        {"role" : "system", "content" : USER_INFO_SUMMARIZER},
        {"role" : "user", "content" : userData}
    ]
).choices[0].message.content

main_topic_provider = """
Create five ML project topics based on the information user provide. The topics for your ML coding project should be created based on user's field. The topics of the coding project must be one that can utilize the user's major.\\
Also, You should create topics by referring to and utilizing the contents of [most_similar_content], which is the AI/ML portfolio for each company. In particular, ML topics should be created using specific domains like the pass portfolio. \\
For example, if the field provided by the user is Computer Vision and the user's major is Marketing, you can create the topic {Image-based product recommendation system, brand logo detection and recognition, makeup product recommendation using virtual trial technology}.
\\\\
You are to provide your answer in the form of a JSON object, following this format: {'first': 'subject1', 'second': 'subject2',...}.
\\
Be sure to check whether the subject matches the field and major entered by the user.
"""

main_topic_candidates = client.chat.completions.create(
    model = model,
    messages = [
        {"role" : "system", "content" : main_topic_provider},
        {"role" : "user", "content" : SUMMARIZED_USER_info},
        {"role" : "user", "content" : most_similar_content }
    ],
    response_format={'type': 'json_object'}
).choices[0].message.content

# 메인 토픽 5개 생성
main_topic_candidates_dict = json.loads(main_topic_candidates)
#print(main_topic_candidates_dict)

# 주제는 첫 번째로 고정
main_topic = main_topic_candidates_dict['second']

OVERALL_ROADMAP_PROVIDER = """
As a lesson planning expert, you are tasked with creating a personalized learning roadmap for the user based on their [main_topic], desired learning field, planned learning period, current knowledge level, and any additional information they want to include in the curriculum. The roadmap should provide a structured plan for the user to follow in order to achieve their learning goals.
You must create a personalized learning roadmap for the user based on the [main_topic].

Please provide a detailed json object containing the following elements:
1. 'overall_roadmap': This should outline the key topics and areas of study that the user needs to cover.
2. 'monthly_plan': Break down the overall roadmap into monthly plans, detailing the specific learning objectives and activities for each month.


Your answer format should be as following:
{'overall_roadmap': '[Overall roadmap user should study.]',
'monthly_plan': {'month1': 'Things user should study in first month.', 'month2': 'Things user should study in second month.', ...}
}
'monthly_plan' should generate the month number considering the user's learning period.
Ensure that the roadmap is tailored to the user's input and provides a clear and actionable guide for their learning journey. Additionally, include the necessary learning elements that users need to perform for each period to support the roadmap specialized in ML/AI. \\Please Write **in Korean**
"""

MONTHLY_PLAN = client.chat.completions.create(
    model=model,
    messages=[
        {'role': 'system', 'content': OVERALL_ROADMAP_PROVIDER},
        {'role': 'user', 'content': SUMMARIZED_USER_info + main_topic}
    ],
    response_format={'type': 'json_object'}
).choices[0].message.content

MONTHLY_PLAN_DICT = json.loads(MONTHLY_PLAN)


WEEKLY_PLAN_PROVIDER = """
You are now the Study Plan Master! Your task is to create a comprehensive and detailed study plan based on the provided information. You will be given an overall roadmap and a monthly plan user will study. You must use these informations to organize a weekly study plan. The study plan should be broken down by week, with each week outlining the specific elements that the user needs to learn.
Create a detailed plan for each month, mapping out weekly activities for the user. For instance, if you're given a plan for two-month, outline the tasks for each of the 8 weeks. The plan should include specific activities and tasks to be completed each week. Weeks shoud be 4 weeks in each month.

You are to provide your answer in the form of a JSON object, following this format: {"month1" : { "week1": {"Things user should study in this week" : {----}. "Details" : {----}. "Practice code topic" : {----}. "Practice Sources" : {"https://github.com/search?q=[]&type=commits"} }, "week2": {"Things user should study in this week" : {----}. "Details" : {----}. "Practice code topic" : {----}. "Practice Sources" : {"https://...",...}} }, "month2" : {...}}.
Please include the development environment setting part in the initial weeks plan like week 1,2. For example, if python is the process of uploading numpy or various libraries in VScode is included in the weekly plan.

Please ensure that the curriculum is tailored to the specific field of study and takes into account the appropriate learning pace for the user's specified learning period. Additionally, the curriculum should be detailed and comprehensive, providing a clear plan and github link to project plan and reference link to lecture link for continuous learning and skill development.
for example of [Practice Sources], in the case of github link, you must put keyword of this plan in this [] fot this given link example 'https://github.com/search?q=[]&type=commits' or 'https://www.cse.psu.edu/~rtc12/CSE486/' or 'https://cs231n.stanford.edu/'.

Remember to be as specific and clear. Also keep in mind that you are not just summarizing monthly plans. You have to provide more detailed things user should do each week to acheive a monthly goal. Your answer should be **in English**.
"""

WEEKLY_PLAN = client.chat.completions.create(
    messages=[
        {'role': 'user', 'content': MONTHLY_PLAN},
        {'role': 'system', 'content': WEEKLY_PLAN_PROVIDER},
    ],
    model=model,
    response_format={'type': 'json_object'}
).choices[0].message.content

WEEKLY_PLAN_DICT = json.loads(WEEKLY_PLAN)

# 위클리 플랜에서 학습 주제, 실습 코드 토픽, 링크 나누기.
L = list(WEEKLY_PLAN_DICT.values())

weekly_learning_object_list = []
weekly_practice_code_list = []

for i in range(len(L)):
    first_list = []
    second_list = []
    for j in range(4):
        week_num = "week" + str(j+1)
        first_list.append(L[i][week_num]['Things user should study in this week'])
        second_list.append(L[i][week_num]['Practice code topic'])
    weekly_learning_object_list.append(first_list)
    weekly_practice_code_list.append(second_list)
    
# 로드맵 출력
CURRICULUM_ENGLISH = """"""
for key, value in MONTHLY_PLAN_DICT.items():
  if key == 'overall_roadmap':
    CURRICULUM_ENGLISH += f"Overall roadmap : {value} \n"
#  else:
#    CURRICULUM_ENGLISH += f"For {key}, user's monthly plan is {value} \n"

for week, plan in WEEKLY_PLAN_DICT.items():
  CURRICULUM_ENGLISH += f"For {week}, user should study {plan}. \n"


FINAL_CURRICULUM_PROVIDER = """
As a study plan organizer, your task is to convert the user's study plan, presented in sentence format, into structured bullet points for the roadmap, monthly plan, and weekly plan. The content of the learning curriculum must be preserved, and no information should be omitted. You may add more specific details if you believe it is necessary.

Please utilize **markdown formats** to highlight important keywords or headers in your response. Please Write **in Korean**.
"""

FINAL_CURRICULUM = client.chat.completions.create(
    model=model,
    messages=[
        {'role': 'system', 'content': FINAL_CURRICULUM_PROVIDER},
        {'role': 'user', 'content': CURRICULUM_ENGLISH}
    ]
).choices[0].message.content

#print("*****************************************************")
#print("\n최종 결과: \n")
#print(main_topic)
#print(FINAL_CURRICULUM)

result = {
    "projectTitle": main_topic,
    "monthTitle": MONTHLY_PLAN,
    "finalCurriculum": FINAL_CURRICULUM
}

print(json.dumps(result, ensure_ascii=False))

#--------------------------실습 코드 생성----------------------------
model = "gpt-4o"

user_info = SUMMARIZED_USER_info

prompt_for_practice_code = """ "Generate a practice code base on the given code_data.

Here's some example of Practice_Code_Information and practice_code.

Example 1.
    Input : 
        Practice_Code_Information:
            Topic and Purpose : The main goal of the code is to implement and train a simple Neural Network Language Model (NNLM). This model performs the task of predicting the next word in a given sentence.
            User Information : Beginner learning NLP. Familiar with coding languages.

    Output : 
        1) Practice_code : 
            import torch
            import torch.nn as nn
            import torch.optim as optim

            def make_batch():
                input_batch = []
                target_batch = []

                for sen in sentences:
                    word = sen.split()  # Split into words
                    input = [word_dict[n] for n in word[:-1]]  # Use all words except the last one as input
                    target = word_dict[word[-1]]  # Use the last word as the target

                    input_batch.append(input)
                    target_batch.append(target)

                return input_batch, target_batch

            # Define the model class
            class NNLM(nn.Module):
                def __init__(self):
                    super(NNLM, self).__init__()
                    self.C = nn.Embedding(n_class, m)  # Word embedding layer
                    self.H = nn.Linear(n_step * m, n_hidden, bias=False)  # Hidden layer
                    self.d = nn.Parameter(torch.ones(n_hidden))  # Hidden layer bias
                    self.U = nn.Linear(n_hidden, n_class, bias=False)  # Output layer
                    self.W = nn.Linear(n_step * m, n_class, bias=False)  # Output layer
                    self.b = nn.Parameter(torch.ones(n_class))  # Output layer bias

                def forward(self, X):
                    X = self.C(X)  # Convert words to vectors through embedding, X: [batch_size, n_step, m]
                    X = X.view(-1, n_step * m)  # Convert to [batch_size, n_step * m]
                    tanh = torch.tanh(self.d + self.H(X))  # [batch_size, n_hidden]
                    output = self.b + self.W(X) + self.U(tanh)  # [batch_size, n_class]
                    return output

            if __name__ == '__main__':
                # Set hyperparameters
                n_step = 2  # Input sequence length (n-1)
                n_hidden = 2  # Hidden layer size
                m = 2  # Embedding vector size

                # Training sentences
                sentences = ["i like dog", "i love coffee", "i hate milk"]

                # Create vocabulary
                word_list = " ".join(sentences).split()
                word_list = list(set(word_list))  # Remove duplicates
                word_dict = {w: i for i, w in enumerate(word_list)}  # Map words to indices
                number_dict = {i: w for i, w in enumerate(word_list)}  # Map indices to words
                n_class = len(word_dict)  # Vocabulary size

                model = NNLM()  # Initialize the model

                criterion = nn.CrossEntropyLoss()  # Define the loss function
                optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define the optimizer

                input_batch, target_batch = make_batch()  # Create batch data
                input_batch = torch.LongTensor(input_batch)  # Convert input data to tensor
                target_batch = torch.LongTensor(target_batch)  # Convert target data to tensor

                # Training process
                for epoch in range(5000):
                    optimizer.zero_grad()  # Initialize the optimizer
                    output = model(input_batch)  # Compute model output

                    loss = criterion(output, target_batch)  # Compute the loss
                    if (epoch + 1) % 1000 == 0:  # Print the loss every 1000 epochs
                        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

                    loss.backward()  # Compute gradients through backpropagation
                    optimizer.step()  # Update the optimizer

                # Prediction
                predict = model(input_batch).data.max(1, keepdim=True)[1]  # Select the word index with the highest probability

                # Print results
                print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

        2) What_You_Can_Learn_from_This_Practice_Code : 
            1. Data Preparation :
                Learn how to preprocess text data and convert it into a suitable format for training a neural network.
                Understand the importance of creating batches of input and target data for training.
            2. Model Construction :
                Understand the different components of the NNLM, including embedding layers, hidden layers, and output layers.
                Learn how to define and initialize neural network layers in PyTorch.
            3. Training Process :
                Learn how to define a loss function (CrossEntropyLoss) and an optimizer (Adam).
                Understand the steps involved in the training loop, including forward pass, loss computation, backward pass, and weight updates.
                Learn how to monitor training progress by printing the loss at regular intervals.
            4. Prediction :
                Learn how to use the trained model to make predictions on new data.
                Understand how to interpret the model’s output and convert it back to human-readable format.
                This practice code provides a comprehensive overview of implementing and training a basic NNLM, covering essential aspects of deep learning applied to natural language processing.

            * Caution : 

Example 2.
    Input : 
    Practice_Code_Information :
            Topic and Purpose: Demonstrates how to use a Convolutional Neural Network (TextCNN) to solve a simple text classification problem. Text classification involves predicting the sentiment or classifying the topic of a given sentence. This code specifically addresses sentiment analysis, predicting whether a sentence is positive (1) or negative (0).
            User Information: Beginner learning NLP. Familiar with coding languages.
        
    Output : 
        1) Practice_Code :
            import numpy as np  # Importing NumPy library for numerical computations
            import torch  # Importing PyTorch, a popular deep learning framework
            import torch.nn as nn  # Importing neural network modules from PyTorch
            import torch.optim as optim  # Importing optimization algorithms from PyTorch
            import torch.nn.functional as F  # Importing functional API from PyTorch for operations without parameters

            # Define TextCNN model
            class TextCNN(nn.Module):  # Defining a custom neural network model class inheriting from nn.Module
                def __init__(self):  # Constructor method for initializing the model
                    super(TextCNN, self).__init__()  # Calling the constructor of the parent class
                    # Initializing parameters for the TextCNN model
                    self.num_filters_total = num_filters * len(filter_sizes)  # Calculating the total number of filters
                    self.W = nn.Embedding(vocab_size, embedding_size)  # Initializing the embedding layer
                    self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)  # Initializing the linear layer
                    self.Bias = nn.Parameter(torch.ones([num_classes]))  # Initializing the bias as a learnable parameter
                    # Creating convolutional filters of various sizes
                    self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

                def forward(self, X):  # Forward pass method of the model
                    # Converting words to embedding vectors [batch_size, sequence_length, embedding_size]
                    embedded_chars = self.W(X)
                    # Adding channel dimension [batch_size, 1, sequence_length, embedding_size]
                    embedded_chars = embedded_chars.unsqueeze(1)

                    pooled_outputs = []  # List to store pooled outputs from convolutional layers
                    for i, conv in enumerate(self.filter_list):  # Iterating over convolutional layers
                        h = F.relu(conv(embedded_chars))  # Applying convolution and Rectified Linear Unit (ReLU) activation
                        # Max pooling operation
                        mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
                        # Changing dimension order
                        pooled = mp(h).permute(0, 3, 2, 1)
                        pooled_outputs.append(pooled)  # Appending pooled output to the list

                    # Concatenating pooling results along the filter size dimension
                    h_pool = torch.cat(pooled_outputs, len(filter_sizes))
                    # Flattening the pooled outputs
                    h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
                    # Applying linear layer and adding bias
                    model = self.Weight(h_pool_flat) + self.Bias
                    return model  # Returning the model output

            if __name__ == '__main__':  # Entry point of the program
                # Define hyperparameters and input data
                embedding_size = 2  # Size of embedding vectors
                sequence_length = 3  # Length of input sentences
                num_classes = 2  # Number of classes (positive/negative)
                filter_sizes = [2, 2, 2]  # Window sizes for n-gram convolutions
                num_filters = 3  # Number of filters

                # Training data
                sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
                labels = [1, 1, 1, 0, 0, 0]  # Class labels (1 for positive, 0 for negative)

                # Create word dictionary
                word_list = " ".join(sentences).split()  # Combining and splitting sentences into words
                word_list = list(set(word_list))  # Removing duplicate words
                word_dict = {w: i for i, w in enumerate(word_list)}  # Mapping words to unique indices
                vocab_size = len(word_dict)  # Size of the vocabulary

                model = TextCNN()  # Initializing the TextCNN model

                criterion = nn.CrossEntropyLoss()  # Defining the loss function (Cross Entropy Loss)
                optimizer = optim.Adam(model.parameters(), lr=0.001)  # Initializing the optimizer (Adam optimizer)

                # Prepare input and target data
                inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
                targets = torch.LongTensor([out for out in labels])  # Converting labels to tensor format

                # Training the model
                for epoch in range(5000):  # Looping over epochs for training
                    optimizer.zero_grad()  # Zeroing gradients
                    output = model(inputs)  # Forward pass to get model output

                    # Calculating loss
                    loss = criterion(output, targets)
                    if (epoch + 1) % 1000 == 0:  # Printing loss every 1000 epochs
                        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

                    loss.backward()  # Backpropagation to compute gradients
                    optimizer.step()  # Updating model parameters using optimizer

                # Testing the trained model
                test_text = 'sorry hate you'
                tests = [np.asarray([word_dict[n] for n in test_text.split()])]
                test_batch = torch.LongTensor(tests)

                # Making predictions
                predict = model(test_batch).data.max(1, keepdim=True)[1]
                if predict[0][0] == 0:
                    print(test_text, "is Bad Mean...")  # Printing negative sentiment if prediction is 0
                else:
                    print(test_text, "is Good Mean!!")  # Printing positive sentiment if prediction is 1

        2) What_You_Can_Learn_from_This_Practice_Code :
            1. Text Data Preprocessing: How to convert words into embedding vectors.
            2. TextCNN Structure: How to use CNN to process text data.
            3. Model Training and Prediction: The process of training a model and making predictions on new data.
            4. Using PyTorch: How to implement and train neural networks using PyTorch.

            * Caution : 

    
You must generate your answer **in Korean**. Also, the Practice_code need very specific, detailed comments for all lines. Add any caution that might be occur when using the [Practice_Code].
Your output format must be same as the examples, especially [1) Practice_Code :], [2) What_You_Can_Learn_from_This_Practice_Code :] have to be exactly same. Your [Output] have to start with "1) Practice_code : ~"
"""

prompt_for_generating_blanks = """
    From given practice_code_real, replace all lines which has comments with '----'.
    Every lines must have comments which . Also, your comments must be in Korean.

    Here's some examples.

    Example1 : 

    This is the given practice_code_real.

    Input : 
        # Import necessary libraries
        import numpy as np  # Importing NumPy library for numerical computations
        import torch  # Importing PyTorch, a popular deep learning framework
        import torch.nn as nn  # Importing neural network modules from PyTorch
        import torch.optim as optim  # Importing optimization algorithms from PyTorch
        import torch.nn.functional as F  # Importing functional API from PyTorch for operations without parameters

        # Define TextCNN model
        class TextCNN(nn.Module):  # Defining a custom neural network model class inheriting from nn.Module
            def __init__(self):  # Constructor method for initializing the model
                super(TextCNN, self).__init__()  # Calling the constructor of the parent class
                # Initializing parameters for the TextCNN model
                self.num_filters_total = num_filters * len(filter_sizes)  # Calculating the total number of filters
                self.W = nn.Embedding(vocab_size, embedding_size)  # Initializing the embedding layer
                self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)  # Initializing the linear layer
                self.Bias = nn.Parameter(torch.ones([num_classes]))  # Initializing the bias as a learnable parameter
                # Creating convolutional filters of various sizes
                self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

            def forward(self, X):  # Forward pass method of the model
                # Converting words to embedding vectors [batch_size, sequence_length, embedding_size]
                embedded_chars = self.W(X)
                # Adding channel dimension [batch_size, 1, sequence_length, embedding_size]
                embedded_chars = embedded_chars.unsqueeze(1)

                pooled_outputs = []  # List to store pooled outputs from convolutional layers
                for i, conv in enumerate(self.filter_list):  # Iterating over convolutional layers
                    h = F.relu(conv(embedded_chars))  # Applying convolution and Rectified Linear Unit (ReLU) activation
                    # Max pooling operation
                    mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
                    # Changing dimension order
                    pooled = mp(h).permute(0, 3, 2, 1)
                    pooled_outputs.append(pooled)  # Appending pooled output to the list

                # Concatenating pooling results along the filter size dimension
                h_pool = torch.cat(pooled_outputs, len(filter_sizes))
                # Flattening the pooled outputs
                h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
                # Applying linear layer and adding bias
                model = self.Weight(h_pool_flat) + self.Bias
                return model  # Returning the model output

        if __name__ == '__main__':  # Entry point of the program
            # Define hyperparameters and input data
            embedding_size = 2  # Size of embedding vectors
            sequence_length = 3  # Length of input sentences
            num_classes = 2  # Number of classes (positive/negative)
            filter_sizes = [2, 2, 2]  # Window sizes for n-gram convolutions
            num_filters = 3  # Number of filters

            # Training data
            sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
            labels = [1, 1, 1, 0, 0, 0]  # Class labels (1 for positive, 0 for negative)

            # Create word dictionary
            word_list = " ".join(sentences).split()  # Combining and splitting sentences into words
            word_list = list(set(word_list))  # Removing duplicate words
            word_dict = {w: i for i, w in enumerate(word_list)}  # Mapping words to unique indices
            vocab_size = len(word_dict)  # Size of the vocabulary

            model = TextCNN()  # Initializing the TextCNN model

            criterion = nn.CrossEntropyLoss()  # Defining the loss function (Cross Entropy Loss)
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Initializing the optimizer (Adam optimizer)

            # Prepare input and target data
            inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
            targets = torch.LongTensor([out for out in labels])  # Converting labels to tensor format

            # Training the model
            for epoch in range(5000):  # Looping over epochs for training
                optimizer.zero_grad()  # Zeroing gradients
                output = model(inputs)  # Forward pass to get model output

                # Calculating loss
                loss = criterion(output, targets)
                if (epoch + 1) % 1000 == 0:  # Printing loss every 1000 epochs
                    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

                loss.backward()  # Backpropagation to compute gradients
                optimizer.step()  # Updating model parameters using optimizer

            # Testing the trained model
            test_text = 'sorry hate you'
            tests = [np.asarray([word_dict[n] for n in test_text.split()])]
            test_batch = torch.LongTensor(tests)

            # Making predictions
            predict = model(test_batch).data.max(1, keepdim=True)[1]
            if predict[0][0] == 0:
                print(test_text, "is Bad Mean...")  # Printing negative sentiment if prediction is 0
            else:
                print(test_text, "is Good Mean!!")  # Printing positive sentiment if prediction is 1

                
    And this is the output you should generate.

    Output : 
        # Import necessary libraries
        import numpy as np  # Importing NumPy library for numerical computations
        import torch  # Importing PyTorch, a popular deep learning framework
        import torch.nn as nn  # Importing neural network modules from PyTorch
        import torch.optim as optim  # Importing optimization algorithms from PyTorch
        import torch.nn.functional as F  # Importing functional API from PyTorch for operations without parameters

        # Define TextCNN model
        class TextCNN(nn.Module):  # Defining a custom neural network model class inheriting from nn.Module
            def __init__(self):  # Constructor method for initializing the model
                super(TextCNN, self).__init__()  # Calling the constructor of the parent class
                # Initializing parameters for the TextCNN model
                ----  # Calculating the total number of filters
                ----  # Initializing the embedding layer
                ----  # Initializing the linear layer
                ----  # Initializing the bias as a learnable parameter
                # Creating convolutional filters of various sizes
                ----

            def forward(self, X):  # Forward pass method of the model
                # Converting words to embedding vectors [batch_size, sequence_length, embedding_size]
                ----
                # Adding channel dimension [batch_size, 1, sequence_length, embedding_size]
                ----

                pooled_outputs = []  # List to store pooled outputs from convolutional layers
                for i, conv in enumerate(self.filter_list):  # Iterating over convolutional layers
                    ----  # Applying convolution and Rectified Linear Unit (ReLU) activation
                    # Max pooling operation
                    ----
                    # Changing dimension order
                    ----
                    ----  # Appending pooled output to the list

                # Concatenating pooling results along the filter size dimension
                ----
                # Flattening the pooled outputs
                ----
                # Applying linear layer and adding bias
                ----
                return model  # Returning the model output

        if __name__ == '__main__':  # Entry point of the program
            # Define hyperparameters and input data
            ----  # Size of embedding vectors
            ----  # Length of input sentences
            ----  # Number of classes (positive/negative)
            ----  # Window sizes for n-gram convolutions
            ----  # Number of filters

            # Training data
            sentences = [----] 
            labels = [----]  # Class labels (1 for positive, 0 for negative)

            # Create word dictionary
            ----  # Combining and splitting sentences into words
            ----  # Removing duplicate words
            ----  # Mapping words to unique indices
            ----  # Size of the vocabulary

            model = ----  # Initializing the TextCNN model

            ----  # Defining the loss function (Cross Entropy Loss)
            ----  # Initializing the optimizer (Adam optimizer)

            # Prepare input and target data
            inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
            targets = torch.LongTensor([out for out in labels])  # Converting labels to tensor format

            # Training the model
            for epoch in range(----):  # Looping over epochs for training
                ----  # Zeroing gradients
                ----  # Forward pass to get model output

                # Calculating loss
                loss = criterion(output, targets)
                if (epoch + 1) % 1000 == 0:  # Printing loss every 1000 epochs
                    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

                ----  # Backpropagation to compute gradients
                ----  # Updating model parameters using optimizer

            # Testing the trained model
            test_text = ----  # Write any sentence that you want to test.
            tests = [np.asarray([word_dict[n] for n in test_text.split()])]
            test_batch = torch.LongTensor(tests)

            # Making predictions
            predict = model(test_batch).data.max(1, keepdim=True)[1]
            if predict[0][0] == 0:
                print(test_text, "is Bad Mean...")  # Printing negative sentiment if prediction is 0
            else:
                print(test_text, "is Good Mean!!")  # Printing positive sentiment if prediction is 1
    
    Example 2:

    Input :
        # Import necessary libraries
        import numpy as np  # Import NumPy for numerical operations
        import torch  # Import PyTorch for deep learning
        import torch.nn as nn  # Import PyTorch's neural network module

        # Define special symbols
        # S: Symbol that shows the start of decoding input
        # E: Symbol that shows the end of decoding output
        # P: Symbol used for padding sequences shorter than the maximum sequence length

        # Function to create training batches
        def make_batch():
            input_batch, output_batch, target_batch = [], [], []  # Initialize lists for inputs, outputs, and targets

            for seq in seq_data:  # Iterate over each sequence pair in the dataset
                for i in range(2):  # For both elements in the sequence pair
                    seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))  # Pad sequences with 'P' to ensure they are of length n_step

                input = [num_dic[n] for n in seq[0]]  # Convert input sequence characters to indices
                output = [num_dic[n] for n in ('S' + seq[1])]  # Add 'S' to the start of the output sequence and convert to indices
                target = [num_dic[n] for n in (seq[1] + 'E')]  # Add 'E' to the end of the target sequence and convert to indices

                input_batch.append(np.eye(n_class)[input])  # Convert input indices to one-hot vectors and add to input batch
                output_batch.append(np.eye(n_class)[output])  # Convert output indices to one-hot vectors and add to output batch
                target_batch.append(target)  # Add target indices to target batch (not one-hot)

            # Convert batches to tensors
            return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)

        # Function to create test batch for a given input word
        def make_testbatch(input_word):
            input_batch, output_batch = [], []  # Initialize lists for input and output batches

            input_w = input_word + 'P' * (n_step - len(input_word))  # Pad input word with 'P' to ensure it is of length n_step
            input = [num_dic[n] for n in input_w]  # Convert input word characters to indices
            output = [num_dic[n] for n in 'S' + 'P' * n_step]  # Create output sequence starting with 'S' followed by 'P' padding

            input_batch = np.eye(n_class)[input]  # Convert input indices to one-hot vectors
            output_batch = np.eye(n_class)[output]  # Convert output indices to one-hot vectors

            return torch.FloatTensor(input_batch).unsqueeze(0), torch.FloatTensor(output_batch).unsqueeze(0)  # Add batch dimension

        # Define the Seq2Seq model
        class Seq2Seq(nn.Module):
            def __init__(self):
                super(Seq2Seq, self).__init__()  # Call the initializer of the parent class

                self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # Define the encoder RNN
                self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # Define the decoder RNN
                self.fc = nn.Linear(n_hidden, n_class)  # Define the fully connected layer to map RNN output to class scores

            def forward(self, enc_input, enc_hidden, dec_input):
                enc_input = enc_input.transpose(0, 1)  # Transpose encoder input for RNN compatibility
                dec_input = dec_input.transpose(0, 1)  # Transpose decoder input for RNN compatibility

                _, enc_states = self.enc_cell(enc_input, enc_hidden)  # Run the encoder RNN and get the final hidden state
                outputs, _ = self.dec_cell(dec_input, enc_states)  # Run the decoder RNN using the encoder's final hidden state

                model = self.fc(outputs)  # Apply the fully connected layer to the RNN outputs
                return model  # Return the final output

        if __name__ == '__main__':
            n_step = 5  # Define the number of time steps
            n_hidden = 128  # Define the number of hidden units

            char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']  # Define the character set
            num_dic = {n: i for i, n in enumerate(char_arr)}  # Create a dictionary to map characters to indices
            seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]  # Define the sequence pairs

            n_class = len(num_dic)  # Define the number of classes
            batch_size = len(seq_data)  # Define the batch size

            model = Seq2Seq()  # Initialize the Seq2Seq model

            criterion = nn.CrossEntropyLoss()  # Define the loss function (cross-entropy loss)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Define the optimizer (Adam optimizer)

            input_batch, output_batch, target_batch = make_batch()  # Create the training batches

            for epoch in range(5000):  # Train for 5000 epochs
                hidden = torch.zeros(1, batch_size, n_hidden)  # Initialize the hidden state

                optimizer.zero_grad()  # Zero the gradients
                output = model(input_batch, hidden, output_batch)  # Run the model
                output = output.transpose(0, 1)  # Transpose the output for compatibility with the loss function
                loss = 0  # Initialize the loss
                for i in range(0, len(target_batch)):  # Compute the loss for each sequence in the batch
                    loss += criterion(output[i], target_batch[i])
                if (epoch + 1) % 1000 == 0:  # Print the loss every 1000 epochs
                    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

            # Function to translate a word using the trained model
            def translate(word):
                input_batch, output_batch = make_testbatch(word)  # Create the test batch

                hidden = torch.zeros(1, 1, n_hidden)  # Initialize the hidden state
                output = model(input_batch, hidden, output_batch)  # Run the model
                predict = output.data.max(2, keepdim=True)[1]  # Get the predicted indices
                decoded = [char_arr[i] for i in predict]  # Convert indices to characters
                end = decoded.index('E')  # Find the end of the sequence
                translated = ''.join(decoded[:end])  # Create the translated string

                return translated.replace('P', '')  # Remove padding symbols

            print('test')
            print('man ->', translate('man'))  # Test the translation for 'man'
            print('mans ->', translate('mans'))  # Test the translation for 'mans'
            print('king ->', translate('king'))  # Test the translation for 'king'
            print('black ->', translate('black'))  # Test the translation for 'black'
            print('upp ->', translate('upp'))  # Test the translation for 'upp'

    Output : 
        # Import necessary libraries
        import numpy as np  # Import NumPy for numerical operations
        import torch  # Import PyTorch for deep learning
        import torch.nn as nn  # Import PyTorch's neural network module

        # Define special symbols
        # S: Symbol that shows the start of decoding input
        # E: Symbol that shows the end of decoding output
        # P: Symbol used for padding sequences shorter than the maximum sequence length

        # Function to create training batches
        def make_batch():
            ----  # Initialize lists for inputs, outputs, and targets

            for ---- in ----:  # Iterate over each sequence pair in the dataset
                for i in ----:  # For both elements in the sequence pair
                    ----  # Pad sequences with 'P' to ensure they are of length n_step

                ----  # Convert input sequence characters to indices
                ----  # Add 'S' to the start of the output sequence and convert to indices
                ----  # Add 'E' to the end of the target sequence and convert to indices

                ----  # Convert input indices to one-hot vectors and add to input batch
                ----  # Convert output indices to one-hot vectors and add to output batch
                ----  # Add target indices to target batch (not one-hot)

            # Convert batches to tensors
            return ----

        # Function to create test batch for a given input word
        def make_testbatch(----):
            ----  # Initialize lists for input and output batches

            ----  # Pad input word with 'P' to ensure it is of length n_step
            ----  # Convert input word characters to indices
            ----  # Create output sequence starting with 'S' followed by 'P' padding

            ----  # Convert input indices to one-hot vectors
            ----  # Convert output indices to one-hot vectors

            return ----  # Add batch dimension

        # Define the Seq2Seq model
        class Seq2Seq(nn.Module):
            def __init__(self):
                super(Seq2Seq, self).__init__()  # Call the initializer of the parent class

                self.enc_cell = nn.RNN(input_size=----, hidden_size=----, dropout=----)  # Define the encoder RNN
                self.dec_cell = nn.RNN(input_size=----, hidden_size=----, dropout=----)  # Define the decoder RNN
                self.fc = nn.Linear(----, ----)  # Define the fully connected layer to map RNN output to class scores

            def forward(self, enc_input, enc_hidden, dec_input):
                enc_input = ----  # Transpose encoder input for RNN compatibility
                dec_input = ----  # Transpose decoder input for RNN compatibility

                _, enc_states = ----  # Run the encoder RNN and get the final hidden state
                outputs, _ = ----  # Run the decoder RNN using the encoder's final hidden state

                model = ----  # Apply the fully connected layer to the RNN outputs
                return model  # Return the final output

        if __name__ == '__main__':
            n_step = ----  # Define the number of time steps
            n_hidden = ----  # Define the number of hidden units

            char_arr = [----]  # Define the character set
            num_dic = {----}  # Create a dictionary to map characters to indices
            seq_data = [----]  # Define the sequence pairs

            n_class = ----  # Define the number of classes
            batch_size = ----  # Define the batch size

            model = ----  # Initialize the Seq2Seq model

            criterion = ----  # Define the loss function (cross-entropy loss)
            optimizer = ----  # Define the optimizer (Adam optimizer)

            ----  # Create the training batches

            for epoch in range(----):  # Train for 5000 epochs
                hidden = ----  # Initialize the hidden state

                ----  # Zero the gradients
                output = ----  # Run the model
                output = ---- # Transpose the output for compatibility with the loss function
                loss = 0  # Initialize the loss
                for i in ----:  # Compute the loss for each sequence in the batch
                    ----
                if ----:  # Print the loss every 1000 epochs
                    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
                ----  # Backpropagate the loss
                ----  # Update the model parameters

            # Function to translate a word using the trained model
            def translate(----):
                ----  # Create the test batch

                hidden = ----  # Initialize the hidden state
                output = ----  # Run the model
                predict = ----  # Get the predicted indices
                decoded = ----  # Convert indices to characters
                end = ----  # Find the end of the sequence
                translated = ----  # Create the translated string

                return ----  # Remove padding symbols

            print('test')
            print(----)  # Test the translation for 'man'
            print(----)  # Test the translation for 'mans'
            print(----)  # Test the translation for 'king'
            print(----)  # Test the translation for 'black'
            print(----)  # Test the translation for 'upp'

    
Think step by step to get the differences between [Input] and [Output] in detail. The parts that the user has to write themselves are replaced with "----". The code replaced with "----" must include comments. If comments are missing, generate them.
Generate a changed practice_code_real. You only need to generate [Output]. Your [Output] have to starts with "# Import necessary libraries\n"

"""

# 백틱을 제거하는 함수
def remove_backticks(s):
    if s.startswith('```'):
        s = s[9:].strip()
    elif s.endswith('```\n'):
        s = s[:-5].strip()
    elif s.endswith('```'):
        s = s[:-3].strip()
    return s


files_to_zip = []

# 전체 커리큘럼 텍스트파일
file_name = "Overall Roadmap.txt"

with open(file_name, "w") as file:
    file.write(FINAL_CURRICULUM)

files_to_zip.append(file_name)

# iteration
for i in range(len(L)):
   for j in range(4):
        topic = weekly_practice_code_list[i][j]
        objective = weekly_learning_object_list[i][j]


        # 실습 코드 주제를 구체화하기
        Practice_Code_Information = f"""
            Topic and Purpose : {topic} + {objective},
            User Information : {user_info}
        """


        #정답과 실습코드 두개 출력시키기

        practice_code = client.chat.completions.create(
            model = "gpt-4o",
            messages=[
                {'role': 'system', 'content': prompt_for_practice_code},
                {'role': 'user', 'content': Practice_Code_Information}
            ]
        ).choices[0].message.content

        #print(type(practice_code))

        print("practice_code : \n")
        print(practice_code)


        code1 = practice_code.split('What_You_Can_Learn_from_This_Practice_Code')

        #print(code1)

        code2 = code1[0][:-3]
        #print(code2)

        # 'What_You_Can_Learn_from_This_Practice_Code : \n' + 
        code_explanation = code1[1][2:] 


        practice_code_real = remove_backticks(code2)


        print("practice_code_real : \n")
        print(practice_code_real)


        practice_code_blank_ver = client.chat.completions.create(
            model = "gpt-4o",
            messages=[
                {'role': 'system', 'content': prompt_for_generating_blanks},
                {'role': 'user', 'content': practice_code_real}
            ]
        ).choices[0].message.content


        print("practice_code_blank_ver : \n")
        print(remove_backticks(practice_code_blank_ver))


        #------------------------------데이터 저장---------------

        #저장할 데이터
        result2 = remove_backticks(code_explanation)
        result3 = remove_backticks(practice_code_real)
        result4 = remove_backticks(practice_code_blank_ver)

        # 현재 스크립트 파일의 디렉토리 경로를 가져오기
        #current_directory = os.path.dirname(os.path.abspath(__file__))
        current_directory = os.getcwd()

        # 폴더 만들기
        folder_name = f"month{i+1}_week{j+1}"
        os.makedirs(folder_name, exist_ok=True) 

        folder_name2 = f"All Result"
        folder_name3 = f"All practice code"
        os.makedirs(folder_name2, exist_ok=True)
        os.makedirs(folder_name3, exist_ok=True)

        folder_path = os.path.join(current_directory, folder_name)
        folder_path2 = os.path.join(current_directory, folder_name2)
        folder_path3 = os.path.join(current_directory, folder_name3)

        # 파일 경로를 지정합니다.
        file_name1 = f"result_month{i+1}_week{j+1}.txt"
        file_name2 = f"practice_code_month{i+1}_week{j+1}.py"

        #file_path1 = os.path.join(current_directory, file_name1)
        #file_path2 = os.path.join(current_directory, file_name2)

        # 파일을 쓰기 모드로 엽니다.
        with open(file_name1, "w") as file:
            # 데이터 쓰기
            file.write("학습 코드에 대한 설명\n")
            file.write(result2)
            file.write("\n\n학습 코드 정답\n")
            file.write(result3)

        with open(file_name2, "w") as file:
            file.write(result4)

        shutil.move(file_name1, os.path.join(folder_name, file_name1))
        shutil.move(file_name2, os.path.join(folder_name, file_name2))

        shutil.copy(os.path.join(folder_name, file_name1), os.path.join(folder_name2, file_name1))
        shutil.copy(os.path.join(folder_name, file_name2), os.path.join(folder_name3, file_name2))

        files_to_zip.append(folder_name)

        print(f"month{i+1}_week{j+1} 파일이 저장되었습니다")

        # 실행 결과를 파일에 작성



# 하나의 폴더에 합치기
# 이름이 유일해야함
parent_folder_name = f"Overall Roadmap for {main_topic}"

parent_folder_path = os.path.join(current_directory, parent_folder_name)

# 상위 폴더 생성
os.makedirs(parent_folder_path, exist_ok=True)

files_to_zip.append(folder_name2)
files_to_zip.append(folder_name3)

# 폴더 복사
for folder in files_to_zip:
    folder_path = os.path.join(current_directory, folder)
    if os.path.isdir(folder_path):
        shutil.copytree(folder_path, os.path.join(parent_folder_path, folder))
        print(f'Folder "{folder}" has been copied to "{parent_folder_name}".')
    else:
        print(f'Folder "{folder}" does not exist.')

print(f'All specified folders have been copied to "{parent_folder_name}".')

shutil.move(file_name, os.path.join(parent_folder_name, file_name))

# 폴더를 ZIP 파일로 압축
shutil.make_archive(parent_folder_name, 'zip', parent_folder_path)



credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=scope)

# 드라이브 서비스 생성
drive_service = build('drive', 'v3', credentials=credentials)

def create_folder(folder_name):
    # 새 폴더 생성 요청
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = drive_service.files().create(body=file_metadata,
                                          fields='id').execute()
    print('폴더가 생성되었습니다. 폴더 ID: %s' % folder.get('id'))
    return folder.get('id')

def upload_file(file_path, folder_id):
    # 파일 업로드
    file_metadata = {'name': parent_folder_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype='text/plain')
    file = drive_service.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id').execute()
    print('파일이 업로드되었습니다. 파일 ID: %s' % file.get('id'))

# 폴더 생성
folder_id = create_folder('pcot_result')

# 파일 업로드
upload_file(parent_folder_name, folder_id)
