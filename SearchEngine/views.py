from django.core.paginator import Paginator
from django.shortcuts import render, redirect
import json

from .forms import SearchForm, FilterForm
import SearchEngine.Backend.search as search


# from Backend import
# Create your views here.


def index(request):
    if request.method == "GET":
        form = SearchForm()
        if 'filtering' in request.session:
            del request.session['filtering']
        if 'form_data' in request.session:
            del request.session['form_data']
        request.session.modified = True

        return render(request, "index.html", {'form': form})


def result_view(request):
    if request.method == 'GET':
        form = SearchForm(request.GET)
        filters = ""
        if 'filtering' in request.session and 'form_data' in request.session:
            if len(request.session['form_data']) > 1:
                data = dict()
                for d in request.session['form_data'][1:]:
                    data.update({d.get('name'): d.get('value')})
                filter_form = FilterForm(initial=data)
            else:
                filter_form = FilterForm()
            filters = request.session['filtering']

        else:
            filter_form = FilterForm()

        if not form.is_valid():
            return redirect('/')

        query = str(request.GET.get('query'))
        searchresults, auto_correct, duration = search.search_string(query)

        if not filters:
            results = searchresults
        else:
            results = [document for document in searchresults if document.category in filters]

        paginator = Paginator(results, 20)  # show 20 page
        page = request.GET.get('page')
        page_results = paginator.get_page(page)

        context = {
            "form": form,
            "filter": filter_form,
            "total_results": len(results),
            "results": page_results,
            "auto_correct": auto_correct,
            "duration": duration
        }

    elif request.method == 'POST':
        form = SearchForm(request.GET)
        filter_form = FilterForm(request.POST)

        if not filter_form.is_valid():
            filter_form = FilterForm()

        else:
            request.session['filtering'] = request.POST.getlist('filters[]')
            jlist = json.loads(request.POST.get('form_data[]'))
            request.session['form_data'] = jlist

        query = str(request.GET.get('query'))
        searchresults, auto_correct, duration = search.search_string(query)
        filters = request.POST.getlist('filters[]')

        if not filters:
            results = searchresults
        else:
            results = [document for document in searchresults if document.category in filters]

        paginator = Paginator(results, 20)  # show 20 page
        page = request.GET.get('page')
        page_results = paginator.get_page(page)

        context = {
            "form": form,
            "filter": filter_form,
            "results": page_results,
            "total_results": len(results),
            "auto_correct": auto_correct,
            "duration": duration
        }

    else:
        return redirect('/')

    return render(request, "result.html", context)
