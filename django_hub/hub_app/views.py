from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

def index(request):
    """The public landing page with Login and Register buttons."""
    if request.user.is_authenticated:
        return redirect('hub')
    return render(request, 'index.html')

def register(request):
    """Handles new user sign-ups."""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('hub')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

@login_required
def hub(request):
    """The protected hub page with 9 buttons to the demos."""
    return render(request, 'hub.html')

def auth_check(request):
    """Nginx calls this to verify if the user can access the demos."""
    if request.user.is_authenticated:
        return HttpResponse("OK", status=200)
    return HttpResponse("Unauthorized", status=401)