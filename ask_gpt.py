from utils import *

prompt = "Respond True if the topic(s) mentioned in following paragraph match(es) the topoics in this list in the same order: The impact of social media on communication, The benefits of reading for pleasure;  otherwise respond False: " +\
"The impact of social media on communication, The benefits of reading for pleasure"

_, respond = retrieve_from_openai(prompt, "gpt-3.5-turbo-16k")
print(respond)