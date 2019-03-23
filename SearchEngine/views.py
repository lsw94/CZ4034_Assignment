from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.


def index(request,*args):
    return render(request,"index.html",{})
    #return HttpResponse("Hello world")

