from django import forms


class SearchForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'id': 'search_input',
                                                          'placeholder': 'Search'}), label='')


class FilterForm(forms.Form):
    art = forms.BooleanField(label='Arts & Culture', required=False)
    business = forms.BooleanField(label='Business', required=False)
    comedy = forms.BooleanField(label='Comedy', required=False)
    crime = forms.BooleanField(label='Crime', required=False)
    education = forms.BooleanField(label='Education', required=False)
    entertain = forms.BooleanField(label='Entertainment', required=False)
    environment = forms.BooleanField(label='Environment', required=False)
    foodDrink = forms.BooleanField(required=False,label='Food & Drink')
    healthyLiving = forms.BooleanField(required=False, label='Healthy Living')
    media = forms.BooleanField(required=False, label='Media')
    money = forms.BooleanField(required=False, label='Money')
    politics = forms.BooleanField(required=False, label='Politics')
    religion = forms.BooleanField(required=False, label='Religion')
    science = forms.BooleanField(required=False, label='Science')
    sports = forms.BooleanField(required=False, label='Sports')
    styleBeauty = forms.BooleanField(required=False, label='Style & Beauty')
    tech = forms.BooleanField(required=False, label='Tech')
    travel = forms.BooleanField(required=False, label='Travel')
    wellness = forms.BooleanField(required=False, label='Wellness')
    worldNews = forms.BooleanField(required=False, label='World News')

