import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class RecipeAssistant:
    def __init__(self):
        self.recipes = []
        self.embeddings = None
        self.index = None
        self.load_recipes("yourpdf.pdf")
        self.build_vector_db()

    def load_recipes(self, file_path: str):
        """Extracts text from a PDF file and stores it in the app's recipes."""
        doc = fitz.open(file_path)
        self.recipes = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.recipes.append({"page": page_num + 1, "content": text})
        print("Recipe book processed successfully!")

    def build_vector_db(self):
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.recipes])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_recipes(self, query: str, k: int = 3):
        """Searches for relevant recipes using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.recipes[i]["content"] for i in I[0]]
        return results if results else ["No relevant recipes found."]

assistant = RecipeAssistant()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = "Welcome to Recipe Development Assistant! I'm here to help you develop new and exciting recipes. Whether you're looking for inspiration, need help with ingredient substitutions, or want to understand the science behind cooking, I'm here to assist. Let's create something delicious together!"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = assistant.search_recipes(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant recipes: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=1000,
        stream=True,
        temperature=0.98,
        top_p=0.7,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown("🍽 Recipe Development Assistant")
    gr.Markdown(
        "📝 This chatbot is designed to assist with recipe development and culinary creativity. "
        "Please note that we are not professional chefs, and the use of this chatbot is at your own responsibility."
    )
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["I'm looking for a new dessert recipe."],
            ["Can you suggest a substitution for eggs in baking?"],
            ["How do I make a vegan lasagna?"],
            ["What is the science behind sourdough fermentation?"],
            ["Can you help me understand the Maillard reaction?"]
        ],
        title='Recipe Development Assistant🍽'
    )

if __name__ == "__main__":
    demo.launch()
