import os
from dotenv import load_dotenv

from django.shortcuts import render
from django.http import HttpResponse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

load_dotenv()


# Create your views here.
def home_view(request):
    """
    Renders the home page template.
    Theis page will serve as the chat page
    """

    return render(request, "core/home.html", {})


def ingestion_view(request):
    """
    Handles GET requests to display the ingestion form
    and POST requests to process uploaded PDF files.
    """
    if request.method == "POST":
        # Handle file upload
        if "data_file" in request.FILES:
            uploaded_file = request.FILES["data_file"]

            # Example: Save the file temporarily and process
            file_path = os.path.join(
                "temp", uploaded_file.name
            )  # Define a temporary path

            os.makedirs("temp", exist_ok=True)  # Ensure temp directory exists
            with open(file_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            try:
                # -----------------------------------
                # Impliment langchain specific logic
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                # Split text
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=0,
                )
                text = text_splitter.split_documents(documents=pages)

                # API keys
                google_api_key = os.getenv("GOOGLE_API_KEY")
                supabase_url = os.getenv("SUPABASE_DB_URL")
                supabase_key = os.getenv("SUPABASE_DB_SERVICE_ROLE_KEY")

                # Generate embeddings
                embedding = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-exp-03-07",
                    google_api_key=google_api_key,
                )

                supabase_client: Client = create_client(supabase_url, supabase_key)

                query_name = "match_documents"
                table_name = "documents"

                vector_store = SupabaseVectorStore.from_documents(
                    text,
                    embedding,
                    client=supabase_client,
                    table_name=table_name,
                    query_name=query_name,
                )

            except Exception as error:
                print(error)

            finally:
                print("completed")

            return HttpResponse(
                f"File '{uploaded_file.name}' received successfully!", status=200
            )

        else:
            # No file uploaded
            return HttpResponse("No file was uploaded in the request.", status=400)

    # Handle GET request - render the form
    return render(request, "core/ingestion.html", {})
