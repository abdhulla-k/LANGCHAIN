from django.shortcuts import render


# Create your views here.
def home_view(request):
    """
    Renders the home page template.
    Theis page will serve as the chat page
    """

    return render(request, "core/home.html", {})


def ingestion_view(request):
    """
    Renders the data ingestion page template.
    This page will handle data input/output
    """

    return render(request, "core/ingestion.html")
