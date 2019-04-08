from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.http import HttpResponseRedirect

# from Backend.search import search_string
from .forms import SearchForm, FilterForm
import SearchEngine.Backend.search as search
import time
# from Backend import
# Create your views here.


def index(request):
    if request.method == "GET":
        form = SearchForm()

        return render(request, "index.html", {'form': form})


def result_view(request):
    start = time.time()
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
        results, auto_correct = search.search_string(query)
        context = {
            "form": form,
            "filter": filter_form,
            "results": results,
            "auto_correct": auto_correct
        }

    elif request.method == 'POST':
        form = SearchForm(request.GET)
        filter_form = FilterForm(request.POST)

        if not filter_form.is_valid():
            filter_form = FilterForm()

        query = str(request.GET.get('query'))
        searchresults, auto_correct = search.search_string(query)
        filters = request.POST.getlist('filters[]')
        print(filters)

        if not filters:
            results = searchresults
        else:
            results = [document for document in searchresults if document.category in filters]
            print(results)

        context = {
            "form": form,
            "filter": filter_form,
            "results": results,
            "auto_correct": auto_correct
        }

    else:
        return redirect('/')

    duration = time.time() - start
    print("Search: %.4fs" % duration)
    return render(request, "result.html", context)
