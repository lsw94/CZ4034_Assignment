from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import HttpResponseRedirect

# from Backend.search import search_string
from .forms import SearchForm, FilterForm
# from Backend import
# Create your views here.


def index(request):
    if request.method == "GET":
        form = SearchForm()

    return render(request, "index.html", {'form': form})


def result_view(request):

    if request.method == 'GET':
        form = SearchForm(request.GET)
        filters = FilterForm(request.GET)

        if not form.is_valid():
            return redirect('/')

        query = str(request.GET.get('query'))
        # results = search_string(query)



    # query results to be inserted here

        context = {
            "form": form,
            "filter": filters,
            "results": ["doc 1", "doc 2","doc 3", "doc 4"]
        }
    else:
        return redirect('/')

    return render(request, "result.html", context)
