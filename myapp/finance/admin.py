from django.contrib import admin
from .models import (
    Profile,
    Category,
    Budget,
    Bill,
    Expense,
    IncomeCategory,
    Income,
    InvestmentCategory,
    Investment
)

# Registering each model to Django admin
admin.site.register(Profile)
admin.site.register(Category)
admin.site.register(Budget)
admin.site.register(Bill)
admin.site.register(Expense)
admin.site.register(IncomeCategory)
admin.site.register(Income)
admin.site.register(InvestmentCategory)
admin.site.register(Investment)
