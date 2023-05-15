from main import *

	
if __name__ == '__main__':
    context, question, answer = load_data()
    
    for i in range(0, len(answer)):
        answer[i] = preprocess(answer[i])
        question[i] = preprocess(question[i])
        
    train_data, test_data, validation_data = load_dataset(answer, question)
    
    train_bert2bert(train_data, validation_data)