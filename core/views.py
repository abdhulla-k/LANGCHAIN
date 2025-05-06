import os

from django.shortcuts import render
from django.http import HttpResponse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


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
                    chunk_overlap=50,
                )

                text = text_splitter.split_documents(documents=pages)
                print(len(text))
                print("\n\n-----------------")
                print(text[0])
                print("\n\n-----------------")

            except:
                "error"
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
