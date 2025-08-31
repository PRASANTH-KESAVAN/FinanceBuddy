from django import forms
from django.contrib.auth.models import User
from .models import Profile, Category, Budget, Bill, Expense, Income, IncomeCategory, Investment, InvestmentCategory
from django.contrib.auth import authenticate

class RegisterForm(forms.ModelForm):
    username = forms.CharField(label ="username", max_length= 255, required=True)
    email = forms.EmailField(label ="email", max_length= 255, required=True)
    password = forms.CharField(label ="password", min_length=8, required=True)
    confirm_password = forms.CharField(label ="confirm password", min_length=8, required=True)

    class Meta:
        model =User
        fields = ['username', 'email', 'password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError("Password doesn't match ")
        

class ProfileForm(forms.ModelForm):
    name = forms.CharField(max_length=200, required=True)
    img_url = forms.ImageField(required=False)
  
    income = forms.IntegerField(required= True)
    savings = forms.IntegerField(required= True)
    

    class Meta:
        model=  Profile
        fields = ['name', 'img_url', 'income','savings']


    def save(self, commit = ...):
        profile = super().save(commit)
        cleaned_data = super().clean()
        img_url = cleaned_data.get('img_url')
        
        if img_url:
          profile.img_url = img_url
        else:
          profile.img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/300px-No_image_available.svg.png"


        if commit:
            profile.save()

        return profile



class LoginForm(forms.Form):
    username = forms.CharField(max_length= 255, required= True)
    password = forms.CharField(label ="password", min_length=8, required=True)

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username')
        password = cleaned_data.get('password')

        if username and password:
            user = authenticate(username = username, password = password)
            if user is None :
                raise forms.ValidationError("Invalid credentials")
            

class BudgetForm(forms.ModelForm):
    category = forms.ModelChoiceField(label = "category", queryset=Category.objects.all())
    amount = forms.IntegerField(label ="amount", required=True)

    class Meta:
        model =  Budget
        fields = ['category', 'amount']


class BillForm(forms.ModelForm):
    BILLING_CYCLE_CHOICES = [
        ('one_time', 'One-Time'),
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('bi_weekly', 'Bi-Weekly'),
        ('monthly', 'Monthly'),
        ('bi_monthly', 'Bi-Monthly (Every Two Months)'),
        ('quarterly', 'Quarterly (Every 3 Months)'),
        ('tri_monthly', 'Tri-Monthly (Every 3 Months)'),
        ('semi_annual', 'Semi-Annual (Every 6 Months)'),
        ('annual', 'Annual (Yearly)'),
        ('custom', 'Custom (User Defined)')
    ]
    title = forms.CharField(max_length=255)  # Bill title (e.g., Electricity Bill)
    amount = forms.DecimalField(max_digits=10, decimal_places=2)  # Bill amount
    due_date = forms.DateField()  # Due date of the bill
    billing_cycle = forms.ChoiceField(choices=BILLING_CYCLE_CHOICES)  # Dropdown for billing cycle
    category = forms.ModelChoiceField(label = "category", queryset=Category.objects.all())


  # Billing cycle

    # created_at = forms.DateTimeField(auto_now_add=True)  # Auto timestamp when bill is created
    # updated_at = forms.DateTimeField(auto_now=True)  # Auto timestamp when bill is updated
    # is_paid = forms.BooleanField(default=False)

    class Meta:
        model = Bill
        fields = ['title', 'amount', 'due_date', 'billing_cycle', 'category']



class ExpenseForm(forms.ModelForm):

    type = forms.CharField(max_length=200, initial="Cash")  # Set default to "Cash"
    amount = forms.DecimalField(max_digits=10, decimal_places=2)
    category = forms.ModelChoiceField(label="category", queryset=Category.objects.all())
    date = forms.DateField()
    notes = forms.CharField(max_length=200)

    class Meta:
        model = Expense
        fields = ['type', 'amount', 'category', 'date', 'notes']



class IncomeForm(forms.ModelForm):
    amount = forms.DecimalField(max_digits=10, decimal_places=2)
    category = forms.ModelChoiceField(label = "category", queryset=IncomeCategory.objects.all())
    date = forms.DateField()

    class Meta:
        model = Income
        fields = ['amount', 'category', 'date', 'notes']

class InvestmentForm(forms.ModelForm):
    investment_type= forms.ModelChoiceField(label = "investment" , queryset=InvestmentCategory.objects.all())
    amount = forms.DecimalField(max_digits=10, decimal_places=2)
    date = forms.DateField()
    profit = forms.DecimalField(max_digits=10, decimal_places=2)
    loss = forms.DecimalField(max_digits= 10, decimal_places=2)
    class Meta :
        model = Investment
        fields = ['investment_type', 'amount', 'date', 'profit', 'loss']