from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded")

# text = "Nitrogen deficiency causes yellowing of leaves."

# embedding = model.encode(text)

# print("Embedding generated!")
# print(type(embedding))
# print(len(embedding))
# print(embedding[:5])  

# threshold=0.75
# context = "Nitrogen deficiency causes yellowing of leaves."
# answers = ["Yellow leaves are caused by nitrogen deficiency", "overwatering."]
# ce = model.encode(context)

# for answer in answers:
#     ae = model.encode(answer)
#     cs = np.dot(ce,ae) / (np.linalg.norm(ae) * np.linalg.norm(ce))

#     if cs>threshold:
#         print("Answer is grounded")
#     else:
#         print("Answer is not grounded")


threshold=0.75
context = "Nitrogen deficiency causes yellowing of leaves."
answers = "Yellow leaves may result from nitrogen deficiency. Overwatering can worsen the issue."
ce = model.encode(context)
claims = answers.split(".")

for claim in claims:
    if claim:
        ae = model.encode(claim)
        cs = np.dot(ce,ae) / (np.linalg.norm(ae) * np.linalg.norm(ce))
        print(claim)
        print(cs)
        if cs>threshold:
            print("Answer is grounded")
        else:
            print("Answer is not grounded")
        print()