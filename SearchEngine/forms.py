from django import forms


class SearchForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'id': 'search_input',
                                                          'placeholder': 'Search'}), label='')
