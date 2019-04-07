from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.http import HttpResponseRedirect

# from Backend.search import search_string
from .forms import SearchForm, FilterForm
import SearchEngine.Backend.search as search
# from Backend import
# Create your views here.


def index(request):
    if request.method == "GET":
        form = SearchForm()

        return render(request, "index.html", {'form': form})


def result_view(request):
    if request.method == 'GET':
        form = SearchForm(request.GET)
        filter_form = FilterForm(request.GET)

        if not form.is_valid():
            return redirect('/')

        if not filter_form.is_valid():
            filter_form = FilterForm()

       # filters = request.GET.get('filters')
        print(request.GET)

        query = str(request.GET.get('query'))
        results = search.search_string(query)
        context = {
            "form": form,
            "filter": filter_form,
            "results": results
        }

    elif request.method == 'POST':
        form = SearchForm(request.GET)
        filter_form = FilterForm(request.POST)

        if not filter_form.is_valid():
            filter_form = FilterForm()

        query = str(request.GET.get('query'))
        searchresults = search.search_string(query)
        filters = request.POST.getlist('filters[]')

        if not filters:
            results = searchresults
        else:
            results = [document for document in searchresults if document.category in filters]

        context = {
            "form": form,
            "filter": filter_form,
            "results": results
        }

    else:
        return redirect('/')

    return render(request, "result.html", context)
