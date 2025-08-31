from django.urls import path
from . import views

app_name = "finance"

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.signup, name='signup'),
    path('setup/', views.setup, name='setup'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('profile/', views.profile, name='profile'),

    path('budget/', views.budget, name='budget'),
    path('set_budget/', views.set_budget, name='set_budget'),
    path('bills/', views.bills, name='bills'),
    path('add_bills/', views.add_bills, name='add_bills'),
    path('pay_bill/<int:bill_id>', views.pay_bill, name='pay_bill'),
    path('expense/', views.expense, name='expense'),
    path('add_expense/', views.add_expense, name='add_expense'),
    path('save-confirmed-expense/', views.save_confirmed_expense, name='save_confirmed_expense'),
    # path('add_expense_voice/', views.add_expense_via_voice, name='add_expense_voice'),

    path('scan_bill_ocr/', views.scan_bill_ocr, name='scan_bill'),
path('save_scanned_expense/', views.save_scanned_expense, name='save_scanned_expense'),



    path('voice-expense/', views.add_expense_via_voice, name='voice_expense'),
    path('add_income/', views.add_income, name='add_income'),
    path('investment/', views.investment, name='investment'),
    path('view_investment/', views.view_investment, name='view_investment'),
    path('add_investment/', views.add_investment, name='add_investment'),
    path('manage_investment/<int:investment_id>', views.manage_investment, name='manage_investment'),
    path('insights/', views.insights, name='insights'),
    path('credit_loan/', views.credit_loan, name='credit_loan'),
    path('chat/', views.chatbot_view, name='chatbot'),
    # path('chatbot/', views.chatbot_page, name='chatbot_page'),


    

















    
]

