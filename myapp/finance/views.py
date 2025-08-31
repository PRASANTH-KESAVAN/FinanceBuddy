import logging
from django.shortcuts import redirect, render, get_object_or_404
from django.http import HttpResponse
from django.contrib import messages
from .forms import LoginForm, ProfileForm, RegisterForm, BudgetForm, BillForm, ExpenseForm, IncomeForm, InvestmentForm
from django.contrib.auth import logout as auth_logout, login as auth_login
from django.contrib.auth import authenticate
from .models import Profile, Category, Budget, Bill, Expense, IncomeCategory, Income, Investment, InvestmentCategory
from django.http import JsonResponse
import re
from django.shortcuts import render
from datetime import timedelta
from django.shortcuts import render
from django.utils.timezone import now
from django.db.models import Sum
from .nlp_utils import detect_intent 
import pandas as pd
import plotly.express as px
import plotly.offline as opy
from django.shortcuts import render
from .models import Profile, Budget, Bill, Expense, Category
from django_pandas.io import read_frame
from django.utils.timezone import now
from datetime import date
import tempfile
import speech_recognition as sr
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django_pandas.io import read_frame
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
from django.contrib.auth.decorators import login_required
# from .ml_utils import prepare_lstm_data, build_lstm_model, forecast_expense_lstm, train_anomaly_detection_model_ml
from plotly.offline import plot as plotly_plot
import plotly.graph_objects as go




def index(request):
    return render(request, 'index.html')

def signup(request):
    form = RegisterForm()
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        
        if form.is_valid():
            
            user = form.save( commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request, "Registration successful.")
            return redirect("finance:login")


    return render(request, 'signup.html', {'form': form})

def setup(request):
    profile_exists = Profile.objects.filter(user=request.user).exists()
    logger = logging.getLogger('Testing')
    logger.debug(f"user id{Profile.objects.filter(user=request.user)} is {profile_exists}")
    if profile_exists:
        return redirect("finance:dashboard") 
    form = ProfileForm()
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES)
        if form.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()
            messages.success(request, "Profile created successfully.")
            return redirect("finance:dashboard")

    return render(request, 'setup.html', {'form': form})
   




# @login_required
# def dashboard(request):
#     profile = Profile.objects.get(user=request.user)

#     # Step 1: Get selected month and year from GET parameters or default to current
#     month = int(request.GET.get('month', timezone.now().month))
#     year = int(request.GET.get('year', timezone.now().year))

#     # Step 2: Define time range for selected month
#     start_of_month = datetime(year, month, 1)
#     if month == 12:
#         end_of_month = datetime(year + 1, 1, 1) - pd.Timedelta(seconds=1)
#     else:
#         end_of_month = datetime(year, month + 1, 1) - pd.Timedelta(seconds=1)

#     # Step 3: Query expenses for that month
#     expenses_qs = Expense.objects.filter(
#         user=request.user,
#         date__range=(start_of_month.date(), end_of_month.date())
#     )

#     # Step 4: Convert to DataFrame
#     expenses_df = read_frame(expenses_qs, fieldnames=['category_id', 'amount'])

#     # Step 5: Aggregate spending by category
#     df = expenses_df.groupby('category_id', as_index=False).sum()
#     df.rename(columns={"amount": "Spent"}, inplace=True)

#     # Step 6: Convert category_id to category name
#     df["Category"] = df["category_id"].str.extract(r'\((\d+)\)').astype(int)
#     df["Category"] = [Category.objects.get(id=id).name if Category.objects.filter(id=id).exists() else "Unknown" for id in df["Category"]]

#     # Step 7: Plot the chart
#     fig = px.bar(df, x="Category", y="Spent",
#                  title=f"Spending by Category - {start_of_month.strftime('%B %Y')}",
#                  labels={"Spent": "Amount"},
#                  color_discrete_sequence=["#1f77b4"])

#     chart_html = opy.plot(fig, auto_open=False, output_type="div")
#     if df.empty:
#         chart_html = "<p class='text-center text-gray-600'>No expense data available for this month.</p>"


#     # For toggling between months
#     context = {
#         'profile': profile,
#         'chart': chart_html,
#         'current_month': month,
#         'current_year': year,
#         'prev_month': month - 1 if month > 1 else 12,
#         'prev_year': year if month > 1 else year - 1,
#         'next_month': month + 1 if month < 12 else 1,
#         'next_year': year if month < 12 else year + 1,
#     }

#     return render(request, 'dashboard.html', context)

from django.shortcuts import render
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django_pandas.io import read_frame
from .models import Profile, Expense, Category, Bill
import pandas as pd
import plotly.express as px
import plotly.offline as opy
from datetime import datetime

@login_required
def dashboard(request):
    profile = Profile.objects.get(user=request.user)
      


    # 1. Handle month/year toggling
    month = int(request.GET.get('month', timezone.now().month))
    year  = int(request.GET.get('year',  timezone.now().year))
    start_of_month = datetime(year, month, 1)
    end_of_month = (datetime(year + (month==12), (month % 12) + 1, 1)
                    - pd.Timedelta(seconds=1))

    # 2. Expenses in selected month
    qs = Expense.objects.filter(
        user=request.user,
        date__range=(start_of_month.date(), end_of_month.date())
    ).select_related('category')
    df_exp = read_frame(qs, fieldnames=['category__name','amount'])

    # 3. Aggregate spending by category
    if df_exp.empty:
        df = pd.DataFrame([{'Category': c.name, 'Spent': 0} for c in Category.objects.all()])
    else:
        df = df_exp.groupby('category__name', as_index=False).sum()
        df.rename(columns={'category__name':'Category','amount':'Spent'}, inplace=True)
        # ensure all categories present
        all_cat = pd.DataFrame([c.name for c in Category.objects.all()], columns=['Category'])
        df = all_cat.merge(df, on='Category', how='left').fillna({'Spent':0})

    # 4. Bar chart (already present)
    bar_fig = px.bar(df, x='Category', y='Spent',
                     title=f"Spending by Category — {start_of_month:%B %Y}",
                     labels={'Spent':'Amount'})
    bar_html = opy.plot(bar_fig, auto_open=False, output_type='div')

    # 5. Pie chart for % wise spending
    pie_fig = px.pie(df, names='Category', values='Spent',
                     title=f"Spending Distribution — {start_of_month:%B %Y}")
    pie_html = opy.plot(pie_fig, auto_open=False, output_type='div')

    # 6. Totals and balance
    total_spent = df['Spent'].sum()
    total_income = profile.income or 0
    balance = total_income - total_spent

    # 7. Dynamic bills for current month
    bills_qs = Bill.objects.filter(
        user=request.user,
        due_date__month=month,
        due_date__year=year
    ).order_by('due_date')

    # 8. prev/next month context
    prev_month = month-1 or 12
    prev_year  = year    if month>1 else year-1
    next_month = month%12 + 1
    next_year  = year    if month<12 else year+1
    
    transactions = Expense.objects.filter(
    user=request.user,
    date__year=year,
    date__month=month
).select_related('category').order_by('-date')


    context = {
        'profile': profile,
        'bar_chart': bar_html,
        'pie_chart': pie_html,
        'transactions': transactions,
        'current_month': month,
        'current_year': year,
        'prev_month': prev_month,
        'prev_year': prev_year,
        'next_month': next_month,
        'next_year': next_year,
        'total_income': total_income,
        'total_spent': total_spent,
        'balance': balance,
        'bills': bills_qs,
    }
    return render(request, 'dashboard.html', context)



def login(request):
   form = LoginForm()
   if request.method == "POST":
         form = LoginForm(request.POST)
         if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
               auth_login(request, user)
               messages.success(request,"Login Successful!")
               return redirect('finance:setup')

   return render(request, 'login.html',{'form':form})



def logout(request):
   auth_logout(request)
   messages.success(request,"Logged Out Succussefully")
   return redirect('finance:index')


from datetime import timedelta, date,datetime
from django.db.models import Sum
from django.utils.timezone import now

def get_current_month_budgets(user):
    today = now()
    start_of_month = today.replace(day=1)
    end_of_month = (start_of_month.replace(month=start_of_month.month % 12 + 1, day=1)
                    if start_of_month.month != 12 else 
                    start_of_month.replace(year=start_of_month.year + 1, month=1, day=1))

    budgets = Budget.objects.filter(
        user=user,
        created_at__gte=start_of_month,
        created_at__lt=end_of_month
    )
    return budgets
@login_required
def budget(request):
    budgets = Budget.objects.filter(user=request.user)
    range_days = int(request.GET.get("range", 90))
    view_type = request.GET.get("view", "recommended")  # 'recommended' or 'actual'
    today = datetime.today()
    start_date = today - timedelta(days=range_days)

    user = request.user
    expenses = Expense.objects.filter(user=user, date__gte=start_date)

    categories = Category.objects.all()

    # Recommended
    recommended_data = expenses.values('category__name').annotate(total=Sum('amount'))
    recommended_dict = {x['category__name']: round(x['total'] / (range_days / 30), 2) for x in recommended_data}

    # Actual
    actual_data = Budget.objects.filter(user=user)
    actual_dict = {b.category.name: b.amount for b in actual_data}

    # Unified Category Labels
    all_labels = sorted(set(list(recommended_dict.keys()) + list(actual_dict.keys())))

    recommended_values = [recommended_dict.get(cat, 0) for cat in all_labels]
    actual_values = [actual_dict.get(cat, 0) for cat in all_labels]

    # Pie chart always uses actuals (e.g., actual spending from Expense model)
    pie_data = expenses.values('category__name').annotate(total=Sum('amount'))
    pie_labels = [x['category__name'] for x in pie_data]
    pie_values = [x['total'] for x in pie_data]

    # Line chart (spending over time)
    daily = expenses.values('date').annotate(total=Sum('amount')).order_by('date')
    line_labels = [x['date'].strftime('%Y-%m-%d') for x in daily]
    line_values = [x['total'] for x in daily]


    # Bill alerts
    upcoming_bills = Bill.objects.filter(
        user=request.user,
        due_date__gte=datetime.today(),
        due_date__lte=datetime.today() + timedelta(days=7),
        is_paid=False
    ).order_by('due_date')

    #### BUDGET INSIGHTS #####  

    user = request.user

    # Income & Expense this month
    today = datetime.today()
    month_start = today.replace(day=1)

    incomes = Income.objects.filter(user=user, date__gte=month_start)
    expenses = Expense.objects.filter(user=user, date__gte=month_start)

    total_income = incomes.aggregate(Sum('amount'))['amount__sum'] or 0
    total_expense = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
    savings = total_income - total_expense

    # Top 3 Expense Categories
    top_expenses = (
        expenses.values('category__name')
        .annotate(total=Sum('amount'))
        .order_by('-total')[:3]
    )

    # Budget vs Expense Insights
    budgets = get_current_month_budgets(user)
    budget_insights = []

    for b in budgets:
        spent = expenses.filter(category=b.category).aggregate(Sum('amount'))['amount__sum'] or 0
        b.spent_amount = spent  # ✅ Add this line
        if spent > b.amount:
            budget_insights.append(f"🚨 Over budget in {b.category.name} (Spent ₹{spent}, Budget ₹{b.amount})")
        elif spent < b.amount * 0.5:
            budget_insights.append(f"🟢 You’ve only spent ₹{spent} in {b.category.name}. Great job!")
            
            
    ################# for spending vs budget######################
    # Get current month range
    profile = Profile.objects.get(user=request.user)
    now = timezone.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_of_month = (start_of_month + pd.DateOffset(months=1)) - pd.Timedelta(seconds=1)

    # Filter by current month for Budget and Expense
    budget_qs = Budget.objects.filter(user=request.user, created_at__range=(start_of_month, end_of_month))
    bills_qs = Bill.objects.filter(user=request.user, created_at__range=(start_of_month, end_of_month))  # Assuming `Bill` has created_at
    expenses_qs = Expense.objects.filter(user=request.user, date__range=(start_of_month.date(), end_of_month.date()))  # Using date for Expense model

    # Convert QuerySets to DataFrames
    budget_df = read_frame(budget_qs, fieldnames=['category_id', 'amount'])
    bills_df = read_frame(bills_qs, fieldnames=['category_id', 'amount'])
    expenses_df = read_frame(expenses_qs, fieldnames=['category_id', 'amount'])

    # Rename amount columns
    budget_df.rename(columns={"amount": "Budgeted"}, inplace=True)
    bills_df.rename(columns={"amount": "Spent"}, inplace=True)
    expenses_df.rename(columns={"amount": "Spent"}, inplace=True)

    # Combine bills and expenses spending
    spending_df = pd.concat([bills_df, expenses_df], ignore_index=True).groupby("category_id", as_index=False).sum()

    # Merge budget and spending data
    df = pd.merge(budget_df, spending_df, on="category_id", how="outer").fillna(0)

    # Extract integer ID from category_id and get category names
    df["Category"] = df["category_id"].str.extract(r'\((\d+)\)').astype(int)
    category = [Category.objects.filter(id=id).first().name if Category.objects.filter(id=id).exists() else "Unknown" for id in df["Category"]]
    df["Category"] = category

    # Ensure numeric types
    df["Budgeted"] = pd.to_numeric(df["Budgeted"], errors='coerce')
    df["Spent"] = pd.to_numeric(df["Spent"], errors='coerce')

    # Create the bar chart
    fig = px.bar(df, x="Category", y=["Budgeted", "Spent"],
                 title="Spending vs Budget (Current Month)",
                 barmode="group",
                 labels={"value": "Amount", "variable": "Type"},
                 color_discrete_map={"Budgeted": "blue", "Spent": "red"})

    chart_html = opy.plot(fig, auto_open=False, output_type="div")






    return render(request, 'budget.html', {'budgets':budgets,
         'chart':chart_html,                                  
        'upcoming_bills': upcoming_bills,
        'labels': all_labels,
        'recommended': recommended_values,
        'actual': actual_values,
        'pie_labels': pie_labels,
        'pie_data': pie_values,
        'line_labels': line_labels,
        'line_data': line_values,
        'selected_range': range_days,
        'view_type': view_type,
         "total_income": total_income,
        "total_expense": total_expense,
        "savings": savings,
        "top_expenses": top_expenses,
        "budget_insights": budget_insights,
    })




@login_required

def set_budget(request):
    categories = Category.objects.all()
    form = BudgetForm()
    if request.method== 'POST':
        form = BudgetForm(request.POST)
        if form.is_valid():
            budget = form.save(commit=False)
            budget.user = request.user
            budget.save()
            messages.success(request, "Budget set successfully.")
            return redirect('finance:budget')
    return render(request, 'set_budget.html', {'form':form, 'categories':categories})

@login_required

def bills(request):
    bills = Bill.objects.filter(user=request.user)
    return render(request, 'bills.html', {'bills':bills})

@login_required

def add_bills(request):
    categories = Category.objects.all()
    form = BillForm()
    if request.method == "POST":
        form = BillForm(request.POST)
        if form.is_valid():
            bill = form.save(commit=False)
            bill.user = request.user
            bill.save()
            messages.success(request, "Bill added successfully.")
            return redirect('finance:bills')

    return render(request, 'add_bill.html', {'categories': categories, 'form':form} )

@login_required

def pay_bill(request, bill_id):
    bill = get_object_or_404(Bill, id=bill_id)
    bill.is_paid =True
    bill.save()
    messages.success(request, "Bill paid successfully.")
    return redirect('finance:bills')


 

from plotly.offline import plot as plotly_plot
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from prophet.plot import plot_plotly
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import Expense, Category
# from .ml_utils import prepare_lstm_data, build_lstm_model, forecast_expense_lstm
import logging
from datetime import timedelta

# Set up logging
from plotly.offline import plot as plotly_plot
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from prophet.plot import plot_plotly
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import Expense, Category
# from .ml_utils import prepare_lstm_data, build_lstm_model, forecast_expense_lstm
import logging
from datetime import timedelta

# Set up logging
logger = logging.getLogger(__name__)

# @login_required
# def expense(request):
#     user = request.user
#     expenses_qs = Expense.objects.filter(user=user)

#     # Convert to DataFrame for initial check
#     df = pd.DataFrame.from_records(expenses_qs.values('date', 'amount'))
#     if df.empty:
#         logger.warning(f"No expense data for user {user.id}")
#         return render(request, 'income_expense_tracking.html', {
#             'error': 'No expense data available for forecasting or anomaly detection',
#             'anomalies_exist': False
#         })

#     # Prepare data for LSTM
#     sequence_length = min(10, len(df) // 2)  # Dynamic sequence length
#     X, y, scaler, df_lstm = prepare_lstm_data(expenses_qs, sequence_length=sequence_length)
#     use_lstm = X is not None and len(X) > 0
#     lst ="not using"
#     proph="not using"

#     if use_lstm:
#         lst="using lst"
#         logger.info(f"LSTM data prepared for user {user.id}: {len(X)} sequences, sequence_length={sequence_length}")
#         # Build and train LSTM model
#         model = build_lstm_model((X.shape[1], 1))
#         model.fit(X, y, epochs=20, batch_size=16, verbose=0)

#         # Forecast next 30 days
#         predictions = forecast_expense_lstm(model, X[-1], future_steps=30, scaler=scaler)

#         # Dates for plotting
#         last_date = df_lstm['ds'].max()
#         future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')

#         # Plot LSTM forecast
#         actual_trace = go.Scatter(
#             x=df_lstm['ds'],
#             y=scaler.inverse_transform(df_lstm[['y']]).flatten(),
#             mode='lines',
#             name='Actual'
#         )
#         pred_trace = go.Scatter(
#             x=future_dates,
#             y=predictions,
#             mode='lines+markers',
#             name='Forecast'
#         )
#         layout = go.Layout(
#             title='Expense Forecast (LSTM)',
#             xaxis_title='Date',
#             yaxis_title='Amount'
#         )
#         fig = go.Figure(data=[actual_trace, pred_trace], layout=layout)
#         chart_html = plotly_plot(fig, auto_open=False, output_type='div')
#     else:
#         proph="using prophest"
#         logger.warning(f"Insufficient data for LSTM for user {user.id}. Falling back to Prophet.")
#         # Fallback to Prophet
#         df_prophet = df.groupby('date')['amount'].sum().reset_index()
#         df_prophet.columns = ['ds', 'y']
#         if len(df_prophet) < 2:
#             logger.error(f"Not enough data for Prophet for user {user.id}: {len(df_prophet)} points")
#             return render(request, 'income_expense_tracking.html', {
#                 'error': 'Not enough data to train forecasting model (minimum 2 unique dates required)',
#                 'anomalies_exist': False
#             })

#         model = Prophet()
#         model.fit(df_prophet)
#         future = model.make_future_dataframe(periods=30)
#         forecast = model.predict(future)
#         forecast_fig = plot_plotly(model, forecast)
#         forecast_fig.update_layout(
#             title="Expense Forecast (Next 30 Days)",
#             xaxis_title="Date",
#             yaxis_title="Amount"
#         )
#         chart_html = plotly_plot(forecast_fig, auto_open=False, output_type='div')

#     # Anomaly Detection
#     anomalies_df = train_anomaly_detection_model(user)
#     logger.info(f"Anomaly detection data shape for user {user.id}: {anomalies_df.shape}")

#     if anomalies_df.empty or len(anomalies_df) < 5:
#         logger.warning(f"Insufficient data for anomaly detection for user {user.id}: {len(anomalies_df)} records")
#         return render(request, 'income_expense_tracking.html', {
#             'chart': chart_html,
#             'error': 'Not enough data to detect anomalies (minimum 5 records required)',
#             'anomalies_exist': False
#         })

#     anomalies = anomalies_df[anomalies_df["anomaly"] == "Anomaly"]
#     anomalies_exist = not anomalies.empty
#     logger.info(f"Anomalies detected for user {user.id}: {len(anomalies)}")

#     # Anomaly Visualization
#     scatter = px.scatter(
#         anomalies_df,
#         x="date",
#         y="amount",
#         color="anomaly",
#         title="Detected Anomalies in Expense Data",
#         labels={"amount": "Amount", "date": "Date"}
#     )
#     scatter_html = plotly_plot(scatter, auto_open=False, output_type='div')

#     return render(request, 'income_expense_tracking.html', {
#         'chart': chart_html,
#         'lst':lst,
#         'proph':proph,
#         'scatter_chart': scatter_html,
#         'anomalies': anomalies.to_dict(orient='records'),
#         'anomalies_exist': anomalies_exist
#     })

from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
# from .ml_utils import prepare_lstm_data, build_lstm_model, forecast_expense_lstm
import logging
from calendar import monthrange

logger = logging.getLogger(__name__)

@login_required
def expense(request):
    
    user = request.user
    expenses_qs = Expense.objects.filter(user=user)

    # Convert to DataFrame
    df = pd.DataFrame.from_records(expenses_qs.values('date', 'amount'))
    if df.empty:
        return render(request, 'income_expense_tracking.html', {
            'error': 'Not enough data to predict or detect anomalies',
            'anomalies_exist': False
        })

    # Forecasting Preparation
    df = df.groupby('date')['amount'].sum().reset_index()
    df.columns = ['ds', 'y']

    # Prophet Forecast Model
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Forecast Plot
    forecast_fig = plot_plotly(model, forecast)
    forecast_fig.update_layout(
        title="Expense Forecast (Next 30 Days)",
        xaxis_title="Date", yaxis_title="Amount"
    )
    chart_html = plotly_plot(forecast_fig, auto_open=False, output_type='div')
    
    start_date = pd.to_datetime('2024-01-01')

    # anomalies_df = train_anomaly_detection_model_ml(user, start_date= start_date)
    anomalies_df = train_anomaly_detection_model(user)
    
    logger.info(f"Anomaly detection data shape for user {user.id}: {anomalies_df.shape}")

    if anomalies_df.empty or len(anomalies_df) < 5:
        logger.warning(f"Insufficient data for anomaly detection for user {user.id}: {len(anomalies_df)} records")
        return render(request, 'income_expense_tracking.html', {
            'chart': chart_html,
            'error': 'Not enough data to detect anomalies (minimum 5 records required)',
            'anomalies_exist': False
        })

    anomalies = anomalies_df[anomalies_df["anomaly"] == "Anomaly"]
    anomalies_exist = not anomalies.empty
    logger.info(f"Anomalies detected for user {user.id}: {len(anomalies)}")

    scatter = px.scatter(
        anomalies_df,
        x="date",
        y="amount",
        color="anomaly",
        title="Detected Anomalies in Expense Data",
        labels={"amount": "Amount", "date": "Date"}
    )
    scatter_html = plotly_plot(scatter, auto_open=False, output_type='div')
    
    

    return render(request, 'income_expense_tracking.html', {
        'chart': chart_html,
      
        'scatter_chart': scatter_html,
        'anomalies': anomalies.to_dict(orient='records'),
        'anomalies_exist': anomalies_exist
    })
    

    
    

def get_user_expenses(user):
    expenses = Expense.objects.filter(user=user)
    data = []
    for exp in expenses:
        data.append([exp.amount, exp.category.name, exp.date])
    return data

def preprocess_expense_data(user):
    data = get_user_expenses(user)
    df = pd.DataFrame(data, columns=["amount", "category", "date"])

    if df.empty:
        logger.warning(f"Empty expense data in preprocess_expense_data for user {user.id}")
        return df

    # Convert date to datetime and extract features
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week_day"] = df["date"].dt.weekday

    # Add category-based feature (e.g., number of unique categories)
    df["category_count"] = df.groupby("category")["category"].transform("count")

    # Normalize the amount
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["amount"]])

    logger.info(f"Preprocessed data for user {user.id}: {df.shape}, columns: {df.columns}")
    return df

def train_anomaly_detection_model(user):
    df = preprocess_expense_data(user)

    if df.empty or len(df) < 5:
        logger.warning(f"Insufficient data in train_anomaly_detection_model for user {user.id}: {len(df)} records")
        return pd.DataFrame(columns=["amount", "category", "date", "month", "day", "week_day", "category_count", "scaled_amount", "anomaly"])

    # Use enhanced feature set
    features = df[["scaled_amount", "month", "day", "week_day", "category_count"]]
    model = IsolationForest(contamination=0.05, random_state=42)  # Reduced contamination
    model.fit(features)

    df["anomaly"] = model.predict(features)
    df["anomaly"] = df["anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

    logger.info(f"Anomaly detection features for user {user.id}: {features.describe().to_dict()}")
    return df


    
@login_required

def add_expense(request):
    categories = Category.objects.all()
    form = ExpenseForm()

    if request.method == "POST":
        form = ExpenseForm(request.POST)
        if form.is_valid():
            expense = form.save(commit=False)
            expense.user = request.user  # Associate user with the expense
            expense.save()
            messages.success(request, "Expense added successfully.")
            return redirect('finance:expense')  # Redirect after saving

    return render(request, 'add_transaction.html', {'categories': categories, 'form': form})







def parse_expense_text(text, user, save=False):
    amount_pattern = r"₹?(\d+)"
    amount_match = re.search(amount_pattern, text)
    amount = int(amount_match.group(1)) if amount_match else None

    category = Category.get_category(text)
    expense_type = 'Cash'
    notes = text

    if amount and category:
        if save:
            expense = Expense(user=user, type=expense_type, category=category,
                              amount=amount, date=date.today(), notes=notes)
            expense.save()
        return {
            'amount': amount,
            'category': category.name,
            'notes': notes,
            'date': date.today()
        }
    else:
        return None




@csrf_exempt
def add_expense_via_voice(request):
    if request.method == "POST":
        text = speech_to_text()

        if text:
            parsed = parse_expense_text(text, request.user, save=False)
            if parsed:
                return JsonResponse({
                    "status": "confirm",
                    "message": f"Did you mean to add ₹{parsed['amount']} for {parsed['category']}?",
                    "data": parsed
                })
            else:
                return JsonResponse({"status": "Failed to extract valid expense details"})
        else:
            return JsonResponse({"status": "No speech input detected"})
    return JsonResponse({"status": "GET not supported"})

import json



@csrf_exempt
def save_confirmed_expense(request):
    if request.method == "POST":
        data = json.loads(request.body)
        try:
            category = Category.objects.get(name=data["category"])
            
            # If the date is not provided in the data, set it to today's date
            expense_date = data.get("date", date.today())  # Use today's date if no date is provided
            
            # Create and save expense instance
            expense = Expense.objects.create(
                user=request.user,
                category=category,
                amount=data["amount"],
                date=expense_date,  # Assign the date here
                type="Cash",  # This can be dynamically set if needed
                notes=data.get("notes", "")
            )
            return JsonResponse({"status": "Expense saved successfully!"})
        except Exception as e:
            return JsonResponse({"status": f"Error saving expense: {str(e)}"})





def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your expense...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)  # You can replace this with your custom logic
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
        return None
    except sr.RequestError:
        print("Sorry, there was an error with the speech-to-text service.")
        return None



import tempfile
from .ml_utils import extract_receipt_data, predict_category
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .test_ocr import extract_receipt_data_easyocr
# views.py
import tempfile
from django.views.decorators.csrf import csrf_exempt

import torch

@csrf_exempt
def scan_bill_ocr(request):
    if request.method == 'POST' and request.FILES.get('bill_image'):
        bill_image = request.FILES['bill_image']

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            for chunk in bill_image.chunks():
                temp_img.write(chunk)
            temp_img_path = temp_img.name

        # Use smart OCR extraction
        extracted_data_eas_ocr = extract_receipt_data_easyocr(temp_img_path)
        
        extracted_data = extract_receipt_data(temp_img_path)
        

        # Predict category based on description
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        category = predict_category(extracted_data_eas_ocr["description"], device=device)

        # Predict category based on description
        # category = predict_category(extracted_data["description"])

        structured_data = {
            "amount": extracted_data["amount"],
            "category": category,
            "description": extracted_data_eas_ocr["description"]
        }

        return JsonResponse({'status': 'success', 'data': structured_data})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


# # views.py
# import tempfile
# from django.views.decorators.csrf import csrf_exempt
# from django.http import JsonResponse
# from .test_ocr import extract_receipt_data_easyocr
# from .ml_utils import predict_category
# import torch

# @csrf_exempt
# def scan_bill_ocr(request):
#     if request.method == 'POST' and request.FILES.get('bill_image'):
#         bill_image = request.FILES['bill_image']

#         # Save uploaded image temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
#             for chunk in bill_image.chunks():
#                 temp_img.write(chunk)
#             temp_img_path = temp_img.name

#         # Use smart OCR extraction
#         extracted_data = extract_receipt_data_easyocr(temp_img_path)

#         # Predict category based on description
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         category = predict_category(extracted_data["description"], device=device)

#         structured_data = {
#             "amount": extracted_data["amount"],
#             "category": category,
#             "description": extracted_data["description"]
#         }

#         return JsonResponse({'status': 'success', 'data': structured_data})

#     return JsonResponse({'status': 'error', 'message': 'Invalid request'})




# now this that thresholding method ------method-1 testing-----
'''
by enhancing the image we are trying whether it will extracting it better


'''

# from .ml_utils import extract_receipt_data_advanced, predict_category

# @csrf_exempt
# def scan_bill_ocr(request):
#     if request.method == 'POST' and request.FILES.get('bill_image'):
#         bill_image = request.FILES['bill_image']

#         # Save uploaded image temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
#             for chunk in bill_image.chunks():
#                 temp_img.write(chunk)
#             temp_img_path = temp_img.name

#         # Use new advanced extraction
#         extracted_data = extract_receipt_data_advanced(temp_img_path)
#         if not extracted_data['full_text']:
#             return JsonResponse({'status': 'error', 'message': 'No text detected in image. Try uploading a clearer receipt.'})


#         # Predict category
#         category = predict_category(extracted_data["description"])

#         structured_data = {
#             "amount": extracted_data["amount"],
#             "category": category,
#             "description": extracted_data["description"],
#             "items": extracted_data["items"],
#             "full_text": extracted_data["full_text"],
#         }

#         return JsonResponse({'status': 'success', 'data': structured_data})

#     return JsonResponse({'status': 'error', 'message': 'Invalid request'})



@csrf_exempt
def save_scanned_expense(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            amount = data.get('amount')
            category_name = data.get('category')
            description = data.get('description', '')

            # Validate required fields
            if not amount or not category_name:
                return JsonResponse({'status': 'error', 'message': 'Amount and category are required'}, status=400)

            # Get or create category
            category = Category.get_category(category_name)
            if not category:
                category, _ = Category.objects.get_or_create(name=category_name)

            # Create expense
            expense = Expense.objects.create(
                user=request.user if request.user.is_authenticated else None,
                type='Expense',  # Assuming type is 'Expense' for scanned bills
                category=category,
                amount=float(amount),
                date=date.today(),  # Use current date; adjust if date is provided
                notes=description
            )

            return JsonResponse({'status': f'Expense saved successfully with ID {expense.id}'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid save request'}, status=400)


@login_required
def add_income(request):
    categories = IncomeCategory.objects.all()
    form = IncomeForm()
    if request.method == "POST":
        form = IncomeForm(request.POST)
        if form.is_valid():
            income = form.save(commit=False)
            income.user = request.user
            income.save()
            messages.success(request, "Income added successfully.")
            return redirect('finance:expense')
    return render(request, "add_income.html", {'categories':categories} )

from .ml_utils import analyze_investments, forecast_roi, predict_category, update_investment_effect
@login_required

def investment(request):
    selected_type = request.GET.get('type')  # e.g., 'Mutual Fund'
    investment_categories = InvestmentCategory.objects.all()

    insights = analyze_investments(request.user, selected_type)
    result = forecast_roi(request.user, months_ahead=6, investment_type=selected_type)

    return render(request, 'savings_investment.html', {
        'insights': insights,
        'result': result,
        'categories': investment_categories,
        'selected_type': selected_type
    })
@login_required

def view_investment(request):
    investments = Investment.objects.filter(user=request.user)
    return render(request, 'view_investment.html', {'investments':investments} )
@login_required
def add_investment(request):
    investment_types = InvestmentCategory.objects.all()
    form = InvestmentForm()
    if request.method == "POST":
        form = InvestmentForm(request.POST)
        if form.is_valid():
            investment = form.save(commit=False)
            investment.user = request.user
            investment.save()

            update_investment_effect(request.user, investment, action="add")

            messages.success(request, "Investment added successfully.")
            return redirect('finance:view_investment')
    return render(request, 'add_investment.html', {'investment_types': investment_types})

# @login_required
# def manage_investment(request, investment_id):
#     investment = get_object_or_404(Investment, id=investment_id)
#     investment_types = InvestmentCategory.objects.all()

#     if request.method == "POST":
#         form = InvestmentForm(request.POST, instance=investment)
#         if form.is_valid():
#             # Remove old investment effects
#             update_investment_effect(request.user, investment, action="delete")

#             investment = form.save(commit=False)
#             investment.save()

#             # Add new effects
#             update_investment_effect(request.user, investment, action="add")

#             messages.success(request, "Investment updated successfully.")
#             return redirect('finance:view_investment')
#     else:
#         form = InvestmentForm(instance=investment)

#     return render(request, 'manage_investment.html', {
#         'investment': investment,
#         'investment_types': investment_types
#     })

from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib import messages
from .models import Investment, InvestmentCategory
from .forms import InvestmentForm
from .ml_utils import update_investment_effect  # Assuming this is defined

@login_required
def manage_investment(request, investment_id):
    investment = get_object_or_404(Investment, id=investment_id)
    investment_types = InvestmentCategory.objects.all()

    if request.method == "POST":
        form = InvestmentForm(request.POST, instance=investment)
        if form.is_valid():
            # Remove old investment effects
            update_investment_effect(request.user, investment, action="delete")

            investment = form.save(commit=False)
            investment.save()

            # Add new effects
            update_investment_effect(request.user, investment, action="add")

            messages.success(request, "Investment updated successfully.")
            return redirect('finance:view_investment')
    else:
        form = InvestmentForm(instance=investment)

    return render(request, 'manage_investment.html', {
        'investment': investment,
        'investment_types': investment_types,
        'form': form  # Added form to context
    })



@login_required

def profile(request):
    profile = Profile.objects.get(user=request.user)
    all_expense= Expense.objects.filter(user=request.user)
    income = Income.objects.filter(user=request.user)



    investment_obj =Investment.objects.filter(user=request.user)
    all_investment_loss =  [ investment.loss for investment in investment_obj]
    total_investment_loss=sum(all_investment_loss)

    all_investment_profit =  [ investment.profit for investment in investment_obj]
    total_investment_profit = sum(all_investment_profit)

    all_investment_amount =  [ investment.amount for investment in investment_obj]
    net_investement = sum(all_investment_amount)




    total_expense=[ expense.amount for expense in all_expense]
    total_net_expense=sum(total_expense)

    total_net_worth = get_income(income, investment_obj, profile)


    savings=total_net_worth-total_net_expense- total_investment_loss
    return render(request, 'profile.html', {'profile': profile,
                                             'total_net_worth': total_net_worth, 
                                             'total_net_expense': total_net_expense,
                                               "savings": savings ,
                                               'investment': net_investement,
                                               'total_investment_profit': total_investment_profit,
                                               'incomes':income
                                               })


def get_income(income_obj, investment_obj, profile_obj):
    all_income= [ income.amount for income in income_obj]
    total_income = sum(all_income)
    all_investment =  [ investment.profit for investment in investment_obj]
    total_investment = sum(all_investment)
    total_net_worth = total_income + total_investment + profile_obj.income
    return total_net_worth



def insights(request):
    return render(request, 'full_report_insights.html')

def credit_loan(request):
    return render(request, 'credit_loan.html')





# @login_required

# def chatbot_view(request):
#     response = None
#     if request.method == 'POST':
#         query = request.POST.get('user_query', '').lower()
#         user = request.user
#         intent = detect_intent(query)

#         try:
#             profile = Profile.objects.get(user=user)

#             # Intent-based responses
#             if intent == "get_name":
#                 response = f"Your name is {profile.name}."
#             elif intent == "get_profile_pic":
#                 response = f"<img src='{profile.profile_picture.url}' width='100'>"
#             elif intent == "get_income_savings":
#                 response = f"Your monthly income is ₹{profile.income} and savings are ₹{profile.savings}."
#             elif intent == "get_total_income":
#                 total = Income.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your total income this month is ₹{total}."
#             elif intent == "get_income_sources":
#                 incomes = Income.objects.filter(user=user)
#                 lines = {}
#                 for i in incomes:
#                     lines[i.category.name] = lines.get(i.category, 0) + i.amount
#                 response = "Your income sources:<br>" + "<br>".join([f"{k}: ₹{v}" for k, v in lines.items()])
#             elif intent == "get_food_expense":
#                 amount = Expense.objects.filter(user=user, category__name__icontains='food', date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You’ve spent ₹{amount} on food this month."
#             elif intent == "get_total_spent":
#                 total_spent = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You've spent ₹{total_spent} this month."
#             elif intent == "get_weekly_expenses":
#                 last_week = now().date() - timedelta(days=7)
#                 expenses = Expense.objects.filter(user=user, date__gte=last_week)
#                 if expenses.exists():
#                     lines = [f"{e.category.name}: ₹{e.amount} on {e.date}" for e in expenses]
#                     response = "Here are your expenses for last week:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No expenses recorded in the last week."
#             elif intent == "get_unpaid_bills":
#                 bills = Bill.objects.filter(user=user, is_paid=False)
#                 if bills.exists():
#                     response = "Unpaid bills:<br>" + "<br>".join([f"{b.category.name} - Due on {b.due_date}" for b in bills])
#                 else:
#                     response = "You have no unpaid bills."
#             elif intent == "get_upcoming_bills":
#                 bills = Bill.objects.filter(user=user, due_date__gte=now().date()).order_by('due_date')
#                 response = "Upcoming bills:<br>" + "<br>".join([f"{b.category.name} - Due on {b.due_date}" for b in bills])
#             elif intent == "get_electricity_due":
#                 bill = Bill.objects.filter(user=user, category__name__icontains="electricity").first()
#                 if bill:
#                     response = f"Your electricity bill is due on {bill.due_date}."
#                 else:
#                     response = "No electricity bill found."
#             elif intent == "get_investments":
#                 invs = Investment.objects.filter(user=user)
#                 lines = [f"{i.investment_type.name}: ₹{i.amount}" for i in invs]
#                 response = "Your investments:<br>" + "<br>".join(lines)
#             elif intent == "get_profit_loss":
#                 profit = Investment.objects.filter(user=user).aggregate(Sum('profit'))['profit__sum'] or 0
#                 loss = Investment.objects.filter(user=user).aggregate(Sum('loss'))['loss__sum'] or 0
#                 response = f"Your total profit is ₹{profit} and loss is ₹{loss}."
#             elif intent == "get_expense_categories":
#                 expenses = Expense.objects.filter(user=user)
#                 categories = expenses.values('category__name').distinct()
#                 response = "Your expense categories are:<br>" + "<br>".join([e['category__name'] for e in categories])
#             elif intent == "get_total_expenses":
#                 total_expenses = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your total expenses this month are ₹{total_expenses}."
#             else:
#                 response = "🤖 Sorry, I didn’t understand that. Can you rephrase your question?"

#         except Exception as e:
#             response = f"❌ Error: {e}"

#     return render(request, 'chatbot.html', {'response': response})


from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils.timezone import now
from django.db.models import Sum
from .models import Profile, Income, Expense, Bill, Investment, Category
from .nlp_utils import detect_intent
from datetime import timedelta
import re

# @login_required
# def chatbot_view(request):
#     # Initialize or retrieve session context
#     if 'chat_context' not in request.session:
#         request.session['chat_context'] = {
#             'history': [],  # List of (query, intent, response) tuples
#             'current_intent': None,
#             'slots': {}  # For slot filling (e.g., amount, category for add_expense)
#         }
#     context = request.session['chat_context']

#     response = None
#     if request.method == 'POST':
#         query = request.POST.get('user_query', '').lower()
#         user = request.user
#         intent = detect_intent(query)

#         try:
#             profile = Profile.objects.get(user=user)

#             # Update context
#             context['history'].append((query, intent, None))  # Response will be added later
#             context['current_intent'] = intent

#             # Helper function to extract slots for add_expense
#             def extract_slots(text):
#                 slots = {}
#                 # Extract amount (e.g., ₹500 or 500)
#                 amount_pattern = r"₹?(\d+)"
#                 amount_match = re.search(amount_pattern, text)
#                 if amount_match:
#                     slots['amount'] = int(amount_match.group(1))

#                 # Extract category
#                 category = Category.get_category(text)
#                 if category:
#                     slots['category'] = category.name

#                 return slots

#             # Intent-based responses with context
#             if intent == "get_name":
#                 response = f"Your name is {profile.name}."
#             elif intent == "get_profile_pic":
#                 response = f"<img src='{profile.profile_picture.url}' width='100'>"
#             elif intent == "get_income_savings":
#                 response = f"Your monthly income is ₹{profile.income} and savings are ₹{profile.savings}."
#             elif intent == "get_total_income":
#                 total = Income.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your total income this month is ₹{total}."
#             elif intent == "get_income_sources":
#                 incomes = Income.objects.filter(user=user)
#                 lines = {}
#                 for i in incomes:
#                     lines[i.category.name] = lines.get(i.category.name, 0) + i.amount
#                 response = "Your income sources:<br>" + "<br>".join([f"{k}: ₹{v}" for k, v in lines.items()])
#             elif intent == "get_food_expense":
#                 amount = Expense.objects.filter(user=user, category__name__icontains='food', date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You’ve spent ₹{amount} on food this month."
#             elif intent == "get_total_spent":
#                 total_spent = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You've spent ₹{total_spent} this month."
#             elif intent == "get_weekly_expenses":
#                 last_week = now().date() - timedelta(days=7)
#                 expenses = Expense.objects.filter(user=user, date__gte=last_week)
#                 if expenses.exists():
#                     lines = [f"{e.category.name}: ₹{e.amount} on {e.date}" for e in expenses]
#                     response = "Here are your expenses for last week:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No expenses recorded in the last week."
#             elif intent == "get_unpaid_bills":
#                 bills = Bill.objects.filter(user=user, is_paid=False)
#                 if bills.exists():
#                     response = "Unpaid bills:<br>" + "<br>".join([f"{b.category.name} - Due on {b.due_date}" for b in bills])
#                 else:
#                     response = "You have no unpaid bills."
#             elif intent == "get_upcoming_bills":
#                 bills = Bill.objects.filter(user=user, due_date__gte=now().date()).order_by('due_date')
#                 response = "Upcoming bills:<br>" + "<br>".join([f"{b.category.name} - Due on {b.due_date}" for b in bills])
#             elif intent == "get_electricity_due":
#                 bill = Bill.objects.filter(user=user, category__name__icontains="electricity").first()
#                 if bill:
#                     response = f"Your electricity bill is due on {bill.due_date}."
#                 else:
#                     response = "No electricity bill found."
#             elif intent == "get_investments":
#                 invs = Investment.objects.filter(user=user)
#                 lines = [f"{i.investment_type.name}: ₹{i.amount}" for i in invs]
#                 response = "Your investments:<br>" + "<br>".join(lines)
#             elif intent == "get_profit_loss":
#                 profit = Investment.objects.filter(user=user).aggregate(Sum('profit'))['profit__sum'] or 0
#                 loss = Investment.objects.filter(user=user).aggregate(Sum('loss'))['loss__sum'] or 0
#                 response = f"Your total profit is ₹{profit} and loss is ₹{loss}."
#             elif intent == "get_expense_categories":
#                 expenses = Expense.objects.filter(user=user)
#                 categories = expenses.values('category__name').distinct()
#                 response = "Your expense categories are:<br>" + "<br>".join([e['category__name'] for e in categories])
#             elif intent == "get_total_expenses":
#                 total_expenses = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your total expenses this month are ₹{total_expenses}."
#             elif intent == "add_expense":
#                 # Extract slots from the current query
#                 new_slots = extract_slots(query)
#                 context['slots'].update(new_slots)

#                 # Check if all required slots are filled
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     # All slots filled, confirm and save expense
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = Category.objects.get(name=category_name)
#                         expense = Expense.objects.create(
#                             user=user,
#                             type='Cash',
#                             category=category,
#                             amount=amount,
#                             date=now().date(),
#                             notes=query
#                         )
#                         response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                         # Clear slots and reset intent
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except Category.DoesNotExist:
#                         response = f"Category '{category_name}' not found. Please choose a valid category (e.g., Food, Utilities)."
#                 else:
#                     # Prompt for missing slots
#                     if 'amount' not in context['slots']:
#                         response = "Please provide the amount for the expense (e.g., ₹500)."
#                     elif 'category' not in context['slots']:
#                         response = "Please provide the category for the expense (e.g., Food, Transportation)."
#             else:
#                 # Handle unknown intent or context-based clarification
#                 if context['history'] and context['history'][-2][1] in ["get_total_spent", "get_total_expenses"]:
#                     # Example: User asked about total spending, now clarify time period
#                     if "last month" in query:
#                         last_month = now().replace(day=1) - timedelta(days=1)
#                         total_spent = Expense.objects.filter(
#                             user=user,
#                             date__month=last_month.month,
#                             date__year=last_month.year
#                         ).aggregate(Sum('amount'))['amount__sum'] or 0
#                         response = f"You spent ₹{total_spent} last month."
#                     elif "this year" in query:
#                         total_spent = Expense.objects.filter(
#                             user=user,
#                             date__year=now().year
#                         ).aggregate(Sum('amount'))['amount__sum'] or 0
#                         response = f"You spent ₹{total_spent} this year."
#                     else:
#                         response = "Could you clarify the time period (e.g., last month, this year)?"
#                 elif context['current_intent'] == "add_expense":
#                     # Handle follow-up for add_expense if user provides partial info
#                     new_slots = extract_slots(query)
#                     context['slots'].update(new_slots)
#                     if 'amount' in context['slots'] and 'category' in context['slots']:
#                         amount = context['slots']['amount']
#                         category_name = context['slots']['category']
#                         try:
#                             category = Category.objects.get(name=category_name)
#                             expense = Expense.objects.create(
#                                 user=user,
#                                 type='Cash',
#                                 category=category,
#                                 amount=amount,
#                                 date=now().date(),
#                                 notes=query
#                             )
#                             response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                             context['slots'] = {}
#                             context['current_intent'] = None
#                         except Category.DoesNotExist:
#                             response = f"Category '{category_name}' not found. Please choose a valid category (e.g., Food, Utilities)."
#                     elif 'amount' not in context['slots']:
#                         response = "Please provide the amount for the expense (e.g., ₹500)."
#                     elif 'category' not in context['slots']:
#                         response = "Please provide the category for the expense (e.g., Food, Transportation)."
#                 else:
#                     response = "🤖 Sorry, I didn’t understand that. Can you rephrase your question?"

#             # Update response in history
#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True  # Ensure session is saved

#         except Exception as e:
#             response = f"❌ Error: {e}"
#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True

#     return render(request, 'chatbot.html', {'response': response})


from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils.timezone import now
from django.db.models import Sum
from .models import Profile, Income, Expense, Bill, Investment, Category, Budget, IncomeCategory, InvestmentCategory
from .nlp_utils import detect_intent
from datetime import timedelta, datetime, date
import re
import pandas as pd

# @login_required
# def chatbot_view(request):
#     # Initialize or retrieve session context
#     if 'chat_context' not in request.session:
#         request.session['chat_context'] = {
#             'history': [],  # List of (query, intent, response) tuples
#             'current_intent': None,
#             'slots': {}  # For slot filling (e.g., amount, category, date)
#         }
#     context = request.session['chat_context']

#     response = None
#     if request.method == 'POST':
#         query = request.POST.get('user_query', '').lower()
#         user = request.user
#         intent = detect_intent(query)
#         print(f"Query: {query} | Intent: {intent} | Confidence: {confidence:.3f}")
        

#         try:
#             profile = Profile.objects.get(user=user)

#             # Update context
#             context['history'].append((query, intent, None))
#             context['current_intent'] = intent

#             # Helper function to extract slots for add intents
#             def extract_slots(text, intent_prefix):
#                 slots = {}
#                 # Extract amount (e.g., ₹500 or 500)
#                 amount_pattern = r"₹?(\d+\.?\d*)"
#                 amount_match = re.search(amount_pattern, text)
#                 if amount_match:
#                     slots['amount'] = float(amount_match.group(1))

#                 # Extract category based on intent prefix
#                 if intent_prefix in ['expense', 'budget', 'bill']:
#                     category = Category.get_category(text)
#                     if category:
#                         slots['category'] = category.name
#                 elif intent_prefix == 'income':
#                     category = IncomeCategory.objects.filter(name__icontains=text.split()[-1]).first()
#                     if category:
#                         slots['category'] = category.name
#                 elif intent_prefix == 'investment':
#                     category = InvestmentCategory.objects.filter(name__icontains=text.split()[-1]).first()
#                     if category:
#                         slots['category'] = category.name

#                 # Extract date (e.g., "next week", "May 15")
#                 date_pattern = r"(next week|this month|last month|on (\w+ \d+)|(\d{4}-\d{2}-\d{2}))"
#                 date_match = re.search(date_pattern, text)
#                 if date_match:
#                     if date_match.group(1) == "next week":
#                         slots['date'] = (now().date() + timedelta(days=7)).strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "this month":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "last month":
#                         slots['date'] = (now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
#                     elif date_match.group(2):
#                         slots['date'] = pd.to_datetime(date_match.group(2), errors='coerce').strftime('%Y-%m-%d')
#                     elif date_match.group(3):
#                         slots['date'] = date_match.group(3)

#                 # Extract billing cycle for bills
#                 if intent_prefix == 'bill':
#                     cycle_pattern = r"(one[-_\s]?time|daily|weekly|bi[-_\s]?weekly|monthly|quarterly|annual)"
#                     cycle_match = re.search(cycle_pattern, text)
#                     if cycle_match:
#                         cycle = cycle_match.group(1).replace(' ', '_').replace('-', '_')
#                         slots['billing_cycle'] = cycle if cycle in [choice[0] for choice in Bill.BILLING_CYCLE_CHOICES] else 'monthly'

#                 return slots

#             # Intent-based responses
#             # Profile Intents
#             if intent == "profile_get_info":
#                 response = f"Your profile: Name: {profile.name}, Income: ₹{profile.income}, Savings: ₹{profile.savings}."
#             elif intent == "profile_get_image":
#                 response = f"<img src='{profile.formatted_img_url}' width='100'>"
#             elif intent == "profile_get_income":
#                 response = f"Your monthly income is ₹{profile.income}."
#             elif intent == "profile_get_savings":
#                 response = f"Your savings are ₹{profile.savings}."
#             elif intent == "profile_get_income_savings":
#                 response = f"Income: ₹{profile.income}, Savings: ₹{profile.savings}."
#             elif intent == "profile_get_slug":
#                 response = f"Your profile slug is {profile.slug}."
#             elif intent == "profile_update_name":
#                 response = "Please provide the new name for your profile."
#                 context['slots']['awaiting'] = 'name'
#             elif intent == "profile_update_image":
#                 response = "Please upload a new profile picture via the profile settings page."
#             elif intent == "profile_update_income":
#                 response = "Please provide the new income amount (e.g., ₹50000)."
#                 context['slots']['awaiting'] = 'income'
#             elif intent == "profile_update_savings":
#                 response = "Please provide the new savings amount (e.g., ₹10000)."
#                 context['slots']['awaiting'] = 'savings'

#             # Category Intents
#             elif intent == "category_get_list":
#                 categories = Category.objects.all()
#                 response = "Expense categories: " + ", ".join([c.name for c in categories])
#             elif intent == "category_check":
#                 category_name = query.split()[-1].capitalize()
#                 exists = Category.objects.filter(name=category_name).exists()
#                 response = f"{category_name} is {'a valid' if exists else 'not a valid'} expense category."
#             elif intent == "category_map":
#                 category = Category.get_category(query)
#                 response = f"The category for '{query}' is {category.name}." if category else "No matching category found."
#             elif intent == "category_add":
#                 response = "Please provide the name of the new category (e.g., Travel)."
#                 context['slots']['awaiting'] = 'category_name'

#             # Budget Intents
#             elif intent == "budget_get":
#                 new_slots = extract_slots(query, 'budget')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     budget = Budget.objects.filter(user=user, category__name=category_name).first()
#                     response = f"Your budget for {category_name} is ₹{budget.amount}." if budget else f"No budget set for {category_name}."
#                 else:
#                     response = "Please specify the category for the budget (e.g., Food)."
#             elif intent == "budget_get_all":
#                 budgets = Budget.objects.filter(user=user)
#                 if budgets:
#                     lines = [f"{b.category.name}: ₹{b.amount}" for b in budgets]
#                     response = "Your budgets:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No budgets set."
#             elif intent == "budget_add":
#                 new_slots = extract_slots(query, 'budget')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = Category.objects.get(name=category_name)
#                         Budget.objects.create(user=user, category=category, amount=amount)
#                         response = f"Budget of ₹{amount} for {category_name} set successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except Category.DoesNotExist:
#                         response = f"Category '{category_name}' not found."
#                 else:
#                     response = "Please provide the amount and category (e.g., ₹500 for Food)."
#             elif intent == "budget_update":
#                 new_slots = extract_slots(query, 'budget')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     budget = Budget.objects.filter(user=user, category__name=category_name).first()
#                     if budget:
#                         budget.amount = amount
#                         budget.save()
#                         response = f"Budget for {category_name} updated to ₹{amount}."
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     else:
#                         response = f"No budget found for {category_name}."
#                 else:
#                     response = "Please provide the new amount and category."
#             elif intent == "budget_check_remaining":
#                 new_slots = extract_slots(query, 'budget')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     budget = Budget.objects.filter(user=user, category__name=category_name).first()
#                     if budget:
#                         spent = Expense.objects.filter(user=user, category__name=category_name, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                         remaining = budget.amount - spent
#                         response = f"You have ₹{remaining} left in your {category_name} budget."
#                     else:
#                         response = f"No budget set for {category_name}."
#                 else:
#                     response = "Please specify the category."
#             elif intent == "budget_get_total":
#                 budgets = Budget.objects.filter(user=user)
#                 total = sum(b.amount for b in budgets)
#                 response = f"Your total budget across all categories is ₹{total}."
#             elif intent == "budget_get_by_date":
#                 start_date = now().date() - timedelta(days=30) if "last month" in query else now().date()
#                 budgets = Budget.objects.filter(user=user, created_at__gte=start_date)
#                 if budgets:
#                     lines = [f"{b.category.name}: ₹{b.amount} (Created: {b.created_at.date()})" for b in budgets]
#                     response = "Budgets:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No budgets found for the specified period."
#             elif intent == "budget_check_exists":
#                 new_slots = extract_slots(query, 'budget')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     exists = Budget.objects.filter(user=user, category__name=category_name).exists()
#                     response = f"A budget for {category_name} {'exists' if exists else 'does not exist'}."
#                 else:
#                     response = "Please specify the category."

#             # Bill Intents
#             elif intent == "bill_get_upcoming":
#                 bills = Bill.objects.filter(user=user, due_date__gte=now().date()).order_by('due_date')
#                 if bills:
#                     lines = [f"{b.title} ({b.category.name}): ₹{b.amount} due on {b.due_date}" for b in bills]
#                     response = "Upcoming bills:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No upcoming bills."
#             elif intent == "bill_get_unpaid":
#                 bills = Bill.objects.filter(user=user, is_paid=False)
#                 if bills:
#                     lines = [f"{b.title} ({b.category.name}): ₹{b.amount} due on {b.due_date}" for b in bills]
#                     response = "Unpaid bills:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No unpaid bills."
#             elif intent == "bill_get_electricity_due":
#                 bill = Bill.objects.filter(user=user, category__name__icontains="electricity").first()
#                 response = f"Your electricity bill is due on {bill.due_date} for ₹{bill.amount}." if bill else "No electricity bill found."
#             elif intent == "bill_add":
#                 new_slots = extract_slots(query, 'bill')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots'] and 'date' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     due_date = context['slots']['date']
#                     billing_cycle = context['slots'].get('billing_cycle', 'monthly')
#                     try:
#                         category = Category.objects.get(name=category_name)
#                         Bill.objects.create(
#                             user=user,
#                             category=category,
#                             title=f"{category_name} Bill",
#                             amount=amount,
#                             due_date=due_date,
#                             billing_cycle=billing_cycle
#                         )
#                         response = f"Bill of ₹{amount} for {category_name} due on {due_date} added!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except Category.DoesNotExist:
#                         response = f"Category '{category_name}' not found."
#                 else:
#                     response = "Please provide the amount, category, and due date (e.g., ₹500 for Electricity due on May 15)."
#             elif intent == "bill_check_status":
#                 new_slots = extract_slots(query, 'bill')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     bill = Bill.objects.filter(user=user, category__name=category_name, is_paid=False).first()
#                     response = f"Your {category_name} bill is {'paid' if bill and bill.is_paid else 'unpaid'}." if bill else f"No {category_name} bill found."
#                 else:
#                     response = "Please specify the bill category."
#             elif intent == "bill_get_total":
#                 bills = Bill.objects.filter(user=user, due_date__month=now().month)
#                 total = sum(b.amount for b in bills)
#                 response = f"Your total bill amount this month is ₹{total}."
#             elif intent == "bill_get_cycle":
#                 new_slots = extract_slots(query, 'bill')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     bill = Bill.objects.filter(user=user, category__name=category_name).first()
#                     response = f"The billing cycle for {category_name} is {bill.billing_cycle}." if bill else f"No {category_name} bill found."
#                 else:
#                     response = "Please specify the bill category."
#             elif intent == "bill_update_status":
#                 new_slots = extract_slots(query, 'bill')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     bill = Bill.objects.filter(user=user, category__name=category_name, is_paid=False).first()
#                     if bill:
#                         bill.is_paid = True
#                         bill.save()
#                         response = f"{category_name} bill marked as paid."
#                     else:
#                         response = f"No unpaid {category_name} bill found."
#                 else:
#                     response = "Please specify the bill category."
#             elif intent == "bill_get_paid":
#                 bills = Bill.objects.filter(user=user, is_paid=True, due_date__month=now().month)
#                 if bills:
#                     lines = [f"{b.title} ({b.category.name}): ₹{b.amount} paid" for b in bills]
#                     response = "Paid bills:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No paid bills this month."

#             # Expense Intents
#             elif intent == "expense_get_by_category":
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     expenses = Expense.objects.filter(user=user, category__name=category_name, date__month=now().month)
#                     total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"You spent ₹{total} on {category_name} this month."
#                 else:
#                     response = "Please specify the expense category."
#             elif intent == "expense_add":
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = Category.objects.get(name=category_name)
#                         Expense.objects.create(
#                             user=user,
#                             type='Cash',
#                             category=category,
#                             amount=amount,
#                             date=now().date(),
#                             notes=query
#                         )
#                         response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except Category.DoesNotExist:
#                         response = f"Category '{category_name}' not found."
#                 else:
#                     response = "Please provide the amount and category (e.g., ₹500 for Food)."
#             elif intent == "expense_get_total":
#                 total = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your total expenses this month are ₹{total}."
#             elif intent == "expense_get_by_date":
#                 start_date = now().date() - timedelta(days=7) if "last week" in query else now().date() - timedelta(days=30)
#                 expenses = Expense.objects.filter(user=user, date__gte=start_date)
#                 if expenses:
#                     lines = [f"{e.category.name}: ₹{e.amount} on {e.date}" for e in expenses]
#                     response = "Expenses:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No expenses found for the specified period."
#             elif intent == "expense_get_by_category_date":
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     start_date = now().date() - timedelta(days=30) if "last month" in query else now().date()
#                     expenses = Expense.objects.filter(user=user, category__name=category_name, date__gte=start_date)
#                     total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"You spent ₹{total} on {category_name} since {start_date}."
#                 else:
#                     response = "Please specify the expense category."
#             elif intent == "expense_get_trend":
#                 expenses = Expense.objects.filter(user=user, date__gte=now().date() - timedelta(days=90))
#                 if expenses:
#                     df = pd.DataFrame.from_records(expenses.values('date', 'amount'))
#                     df = df.groupby('date')['amount'].sum().reset_index()
#                     trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                     response = f"Your average daily spending trend over the last 90 days is ₹{trend:.2f}."
#                 else:
#                     response = "No expense data available for trend analysis."
#             elif intent == "expense_get_trend_by_category":
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     expenses = Expense.objects.filter(user=user, category__name=category_name, date__gte=now().date() - timedelta(days=90))
#                     if expenses:
#                         df = pd.DataFrame.from_records(expenses.values('date', 'amount'))
#                         df = df.groupby('date')['amount'].sum().reset_index()
#                         trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                         response = f"Your average daily spending trend for {category_name} is ₹{trend:.2f}."
#                     else:
#                         response = f"No {category_name} expenses found for trend analysis."
#                 else:
#                     response = "Please specify the expense category."
#             elif intent == "expense_get_max":
#                 expenses = Expense.objects.filter(user=user, date__month=now().month)
#                 if expenses:
#                     max_expense = expenses.order_by('-amount').first()
#                     response = f"Your highest expense this month was ₹{max_expense.amount} on {max_expense.category.name}."
#                 else:
#                     response = "No expenses recorded this month."
#             elif intent == "expense_get_average":
#                 expenses = Expense.objects.filter(user=user, date__gte=now().date() - timedelta(days=365))
#                 if expenses:
#                     avg = expenses.aggregate(Sum('amount'))['amount__sum'] / 12
#                     response = f"Your average monthly spending is ₹{avg:.2f}."
#                 else:
#                     response = "No expense data available."
#             elif intent == "expense_get_average_by_category":
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     expenses = Expense.objects.filter(user=user, category__name=category_name, date__gte=now().date() - timedelta(days=365))
#                     if expenses:
#                         avg = expenses.aggregate(Sum('amount'))['amount__sum'] / 12
#                         response = f"Your average monthly spending on {category_name} is ₹{avg:.2f}."
#                     else:
#                         response = f"No {category_name} expenses found."
#                 else:
#                     response = "Please specify the expense category."

#             # IncomeCategory Intents
#             elif intent == "income_category_get_list":
#                 categories = IncomeCategory.objects.all()
#                 response = "Income categories: " + ", ".join([c.name for c in categories])
#             elif intent == "income_category_check":
#                 category_name = query.split()[-1].capitalize()
#                 exists = IncomeCategory.objects.filter(name=category_name).exists()
#                 response = f"{category_name} is {'a valid' if exists else 'not a valid'} income category."
#             elif intent == "income_category_map":
#                 category_name = query.split()[-1].capitalize()
#                 category = IncomeCategory.objects.filter(name__icontains=category_name).first()
#                 response = f"The income category for '{query}' is {category.name}." if category else "No matching income category found."
#             elif intent == "income_category_add":
#                 response = "Please provide the name of the new income category (e.g., Salary)."
#                 context['slots']['awaiting'] = 'income_category_name'

#             # Income Intents
#             elif intent == "income_get_by_category":
#                 new_slots = extract_slots(query, 'income')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     incomes = Income.objects.filter(user=user, category__name=category_name, date__month=now().month)
#                     total = incomes.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"You earned ₹{total} from {category_name} this month."
#                 else:
#                     response = "Please specify the income category."
#             elif intent == "income_add":
#                 new_slots = extract_slots(query, 'income')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = IncomeCategory.objects.get(name=category_name)
#                         Income.objects.create(
#                             user=user,
#                             category=category,
#                             amount=amount,
#                             date=now().date(),
#                             notes=query
#                         )
#                         response = f"Income of ₹{amount} for {category_name} added successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except IncomeCategory.DoesNotExist:
#                         response = f"Income category '{category_name}' not found."
#                 else:
#                     response = "Please provide the amount and category (e.g., ₹5000 for Salary)."
#             elif intent == "income_get_total":
#                 total = Income.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your total income this month is ₹{total}."
#             elif intent == "income_get_by_date":
#                 start_date = now().date() - timedelta(days=7) if "last week" in query else now().date() - timedelta(days=30)
#                 incomes = Income.objects.filter(user=user, date__gte=start_date)
#                 if incomes:
#                     lines = [f"{i.category.name}: ₹{i.amount} on {i.date}" for i in incomes]
#                     response = "Income:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No income recorded for the specified period."
#             elif intent == "income_get_by_category_date":
#                 new_slots = extract_slots(query, 'income')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     start_date = now().date() - timedelta(days=30) if "last month" in query else now().date()
#                     incomes = Income.objects.filter(user=user, category__name=category_name, date__gte=start_date)
#                     total = incomes.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"You earned ₹{total} from {category_name} since {start_date}."
#                 else:
#                     response = "Please specify the income category."
#             elif intent == "income_get_trend":
#                 incomes = Income.objects.filter(user=user, date__gte=now().date() - timedelta(days=90))
#                 if incomes:
#                     df = pd.DataFrame.from_records(incomes.values('date', 'amount'))
#                     df = df.groupby('date')['amount'].sum().reset_index()
#                     trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                     response = f"Your average daily income trend over the last 90 days is ₹{trend:.2f}."
#                 else:
#                     response = "No income data available for trend analysis."
#             elif intent == "income_get_trend_by_category":
#                 new_slots = extract_slots(query, 'income')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     incomes = Income.objects.filter(user=user, category__name=category_name, date__gte=now().date() - timedelta(days=90))
#                     if incomes:
#                         df = pd.DataFrame.from_records(incomes.values('date', 'amount'))
#                         df = df.groupby('date')['amount'].sum().reset_index()
#                         trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                         response = f"Your average daily income trend for {category_name} is ₹{trend:.2f}."
#                     else:
#                         response = f"No {category_name} income found for trend analysis."
#                 else:
#                     response = "Please specify the income category."
#             elif intent == "income_get_max":
#                 incomes = Income.objects.filter(user=user, date__month=now().month)
#                 if incomes:
#                     max_income = incomes.order_by('-amount').first()
#                     response = f"Your highest income this month was ₹{max_income.amount} from {max_income.category.name}."
#                 else:
#                     response = "No income recorded this month."
#             elif intent == "income_get_average":
#                 incomes = Income.objects.filter(user=user, date__gte=now().date() - timedelta(days=365))
#                 if incomes:
#                     avg = incomes.aggregate(Sum('amount'))['amount__sum'] / 12
#                     response = f"Your average monthly income is ₹{avg:.2f}."
#                 else:
#                     response = "No income data available."
#             elif intent == "income_get_average_by_category":
#                 new_slots = extract_slots(query, 'income')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     incomes = Income.objects.filter(user=user, category__name=category_name, date__gte=now().date() - timedelta(days=365))
#                     if incomes:
#                         avg = incomes.aggregate(Sum('amount'))['amount__sum'] / 12
#                         response = f"Your average monthly income from {category_name} is ₹{avg:.2f}."
#                     else:
#                         response = f"No {category_name} income found."
#                 else:
#                     response = "Please specify the income category."

#             # InvestmentCategory Intents
#             elif intent == "investment_category_get_list":
#                 categories = InvestmentCategory.objects.all()
#                 response = "Investment categories: " + ", ".join([c.name for c in categories])
#             elif intent == "investment_category_check":
#                 category_name = query.split()[-1].capitalize()
#                 exists = InvestmentCategory.objects.filter(name=category_name).exists()
#                 response = f"{category_name} is {'a valid' if exists else 'not a valid'} investment category."
#             elif intent == "investment_category_map":
#                 category_name = query.split()[-1].capitalize()
#                 category = InvestmentCategory.objects.filter(name__icontains=category_name).first()
#                 response = f"The investment category for '{query}' is {category.name}." if category else "No matching investment category found."
#             elif intent == "investment_category_add":
#                 response = "Please provide the name of the new investment category (e.g., Stocks)."
#                 context['slots']['awaiting'] = 'investment_category_name'

#             # Investment Intents
#             elif intent == "investment_get_by_category":
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     investments = Investment.objects.filter(user=user, investment_type__name=category_name)
#                     total = investments.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"You invested ₹{total} in {category_name}."
#                 else:
#                     response = "Please specify the investment category."
#             elif intent == "investment_add":
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = InvestmentCategory.objects.get(name=category_name)
#                         Investment.objects.create(
#                             user=user,
#                             investment_type=category,
#                             amount=amount,
#                             date=now().date()
#                         )
#                         response = f"Investment of ₹{amount} in {category_name} added successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except InvestmentCategory.DoesNotExist:
#                         response = f"Investment category '{category_name}' not found."
#                 else:
#                     response = "Please provide the amount and category (e.g., ₹5000 for Stocks)."
#             elif intent == "investment_get_total":
#                 total = Investment.objects.filter(user=user).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your total investment amount is ₹{total}."
#             elif intent == "investment_get_profit":
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     profit = Investment.objects.filter(user=user, investment_type__name=category_name).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your profit from {category_name} is ₹{profit}."
#                 else:
#                     response = "Please specify the investment category."
#             elif intent == "investment_get_loss":
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     loss = Investment.objects.filter(user=user, investment_type__name=category_name).aggregate(Sum('loss'))['loss__sum'] or 0
#                     response = f"Your loss from {category_name} is ₹{loss}."
#                 else:
#                     response = "Please specify the investment category."
#             elif intent == "investment_get_by_date":
#                 start_date = now().date() - timedelta(days=30) if "last month" in query else now().date()
#                 investments = Investment.objects.filter(user=user, date__gte=start_date)
#                 if investments:
#                     lines = [f"{i.investment_type.name}: ₹{i.amount} on {i.date}" for i in investments]
#                     response = "Investments:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No investments found for the specified period."
#             elif intent == "investment_get_performance":
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     investments = Investment.objects.filter(user=user, investment_type__name=category_name)
#                     total_invested = investments.aggregate(Sum('amount'))['amount__sum'] or 0
#                     total_profit = investments.aggregate(Sum('profit'))['profit__sum'] or 0
#                     total_loss = investments.aggregate(Sum('loss'))['loss__sum'] or 0
#                     response = f"For {category_name}: Invested ₹{total_invested}, Profit ₹{total_profit}, Loss ₹{total_loss}."
#                 else:
#                     response = "Please specify the investment category."
#             elif intent == "investment_get_trend":
#                 investments = Investment.objects.filter(user=user, date__gte=now().date() - timedelta(days=90))
#                 if investments:
#                     df = pd.DataFrame.from_records(investments.values('date', 'amount'))
#                     df = df.groupby('date')['amount'].sum().reset_index()
#                     trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                     response = f"Your average daily investment trend is ₹{trend:.2f}."
#                 else:
#                     response = "No investment data available for trend analysis."
#             elif intent == "investment_get_trend_by_category":
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     investments = Investment.objects.filter(user=user, investment_type__name=category_name, date__gte=now().date() - timedelta(days=90))
#                     if investments:
#                         df = pd.DataFrame.from_records(investments.values('date', 'amount'))
#                         df = df.groupby('date')['amount'].sum().reset_index()
#                         trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                         response = f"Your average daily investment trend for {category_name} is ₹{trend:.2f}."
#                     else:
#                         response = f"No {category_name} investments found for trend analysis."
#                 else:
#                     response = "Please specify the investment category."
#             elif intent == "investment_get_max":
#                 investments = Investment.objects.filter(user=user, date__month=now().month)
#                 if investments:
#                     max_investment = investments.order_by('-amount').first()
#                     response = f"Your highest investment this month was ₹{max_investment.amount} in {max_investment.investment_type.name}."
#                 else:
#                     response = "No investments recorded this month."
#             elif intent == "investment_get_max_profit":
#                 investments = Investment.objects.filter(user=user, profit__gt=0)
#                 if investments:
#                     max_profit = investments.order_by('-profit').first()
#                     response = f"Your most profitable investment was ₹{max_profit.profit} from {max_profit.investment_type.name}."
#                 else:
#                     response = "No profitable investments recorded."

#             # Handle slot-filling for updates and additions
#             elif context['slots'].get('awaiting'):
#                 if context['slots']['awaiting'] == 'name':
#                     profile.name = query
#                     profile.save()
#                     response = f"Profile name updated to {query}."
#                     context['slots'] = {}
#                 elif context['slots']['awaiting'] == 'income':
#                     try:
#                         profile.income = float(query)
#                         profile.save()
#                         response = f"Income updated to ₹{query}."
#                         context['slots'] = {}
#                     except ValueError:
#                         response = "Please provide a valid income amount."
#                 elif context['slots']['awaiting'] == 'savings':
#                     try:
#                         profile.savings = float(query)
#                         profile.save()
#                         response = f"Savings updated to ₹{query}."
#                         context['slots'] = {}
#                     except ValueError:
#                         response = "Please provide a valid savings amount."
#                 elif context['slots']['awaiting'] == 'category_name':
#                     Category.objects.get_or_create(name=query.capitalize())
#                     response = f"Category '{query.capitalize()}' added."
#                     context['slots'] = {}
#                 elif context['slots']['awaiting'] == 'income_category_name':
#                     IncomeCategory.objects.get_or_create(name=query.capitalize())
#                     response = f"Income category '{query.capitalize()}' added."
#                     context['slots'] = {}
#                 elif context['slots']['awaiting'] == 'investment_category_name':
#                     InvestmentCategory.objects.get_or_create(name=query.capitalize())
#                     response = f"Investment category '{query.capitalize()}' added."
#                     context['slots'] = {}

#             # Handle context-based clarification
#             else:
#                 if context['history'] and context['history'][-2][1] in ["get_total_spent", "get_total_expenses"]:
#                     if "last month" in query:
#                         last_month = now().replace(day=1) - timedelta(days=1)
#                         total_spent = Expense.objects.filter(
#                             user=user,
#                             date__month=last_month.month,
#                             date__year=last_month.year
#                         ).aggregate(Sum('amount'))['amount__sum'] or 0
#                         response = f"You spent ₹{total_spent} last month."
#                     elif "this year" in query:
#                         total_spent = Expense.objects.filter(
#                             user=user,
#                             date__year=now().year
#                         ).aggregate(Sum('amount'))['amount__sum'] or 0
#                         response = f"You spent ₹{total_spent} this year."
#                     else:
#                         response = "Could you clarify the time period (e.g., last month, this year)?"
#                 else:
#                     response = "🤖 Sorry, I didn’t understand that. Can you rephrase your question?"

#             # Update response in history
#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True

#         except Exception as e:
#             response = f"❌ Error: {e}"
#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True

#     return render(request, 'chatbot.html', {'response': response})


# import logging
# import re
# from datetime import datetime, timedelta
# from django.contrib.auth.decorators import login_required
# from django.shortcuts import render
# from django.utils.timezone import now
# from django.db.models import Sum
# from .models import Profile, Category, Budget, Bill, Expense, Income, IncomeCategory, Investment, InvestmentCategory
# import pandas as pd
# from .nlp_utils import detect_intent

# logger = logging.getLogger(__name__)

# @login_required
# def chatbot_view(request):
#     # Initialize session context
#     if 'chat_context' not in request.session:
#         request.session['chat_context'] = {
#             'history': [],  # (query, intent, response)
#             'current_intent': None,
#             'slots': {}
#         }
#     context = request.session['chat_context']

#     response = None
#     if request.method == 'POST':
#         query = request.POST.get('user_query', '').lower()
#         user = request.user
#         intent = detect_intent(query)
#         logger.info(f"Query: {query} | Intent: {intent}")
#         print(f"Query: {query} | Intent: {intent}")

#         try:
#             profile = Profile.objects.get(user=user)
#             context['history'].append((query, intent, None))
#             context['current_intent'] = intent

#             def extract_slots(text, intent_prefix):
#                 slots = {}
#                 # Extract amount (e.g., ₹500, 500, ₹500.00)
#                 amount_pattern = r"₹?\s*(\d+\.?\d*)"
#                 amount_match = re.search(amount_pattern, text)
#                 if amount_match:
#                     slots['amount'] = float(amount_match.group(1))

#                 # Extract category
#                 category_keywords = {
#                     'expense': ['food', 'dining', 'groceries', 'travel', 'electricity', 'utilities'],
#                     'budget': ['food', 'dining', 'groceries', 'travel'],
#                     'bill': ['electricity', 'utilities', 'internet', 'phone'],
#                     'income': ['salary', 'freelance', 'bonus'],
#                     'investment': ['stocks', 'mutual funds', 'bonds']
#                 }
#                 text_words = text.lower().split()
#                 for keyword in category_keywords.get(intent_prefix, []):
#                     if keyword in text_words:
#                         if intent_prefix in ['expense', 'budget', 'bill']:
#                             category = Category.objects.filter(name__icontains=keyword).first()
#                         elif intent_prefix == 'income':
#                             category = IncomeCategory.objects.filter(name__icontains=keyword).first()
#                         elif intent_prefix == 'investment':
#                             category = InvestmentCategory.objects.filter(name__icontains=keyword).first()
#                         if category:
#                             slots['category'] = category.name
#                             break
#                 # Fallback to last word
#                 if 'category' not in slots and text_words:
#                     last_word = text_words[-1].capitalize()
#                     if intent_prefix in ['expense', 'budget', 'bill']:
#                         category = Category.objects.filter(name__icontains=last_word).first()
#                     elif intent_prefix == 'income':
#                         category = IncomeCategory.objects.filter(name__icontains=last_word).first()
#                     elif intent_prefix == 'investment':
#                         category = InvestmentCategory.objects.filter(name__icontains=last_word).first()
#                     if category:
#                         slots['category'] = category.name

#                 # Extract date
#                 date_pattern = r"(next week|this month|last month|on (\w+ \d+)|(\d{4}-\d{2}-\d{2})|today|tomorrow)"
#                 date_match = re.search(date_pattern, text)
#                 if date_match:
#                     if date_match.group(1) == "next week":
#                         slots['date'] = (now().date() + timedelta(days=7)).strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "this month":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "last month":
#                         slots['date'] = (now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
#                     elif date_match.group(2):
#                         slots['date'] = pd.to_datetime(date_match.group(2), errors='coerce').strftime('%Y-%m-%d')
#                     elif date_match.group(3):
#                         slots['date'] = date_match.group(3)
#                     elif date_match.group(1) == "today":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "tomorrow":
#                         slots['date'] = (now().date() + timedelta(days=1)).strftime('%Y-%m-%d')

#                 logger.info(f"Extracted slots for {intent_prefix}: {slots}")
#                 return slots

#             # Intent-based responses
#             if intent == "get_name":
#                 logger.info(f"Handling get_name for user {user.username}")
#                 response = f"Your name is {profile.name}."
#             elif intent == "get_total_spent":
#                 logger.info(f"Handling get_total_spent for user {user.username}")
#                 total = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You spent ₹{total} this month."
#             elif intent == "get_income_sources":
#                 logger.info(f"Handling get_income_sources for user {user.username}")
#                 categories = IncomeCategory.objects.all()
#                 response = "Income sources: " + ", ".join([c.name for c in categories])
#             elif intent == "get_food_expense":
#                 logger.info(f"Handling get_food_expense for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 category_name = context['slots'].get('category', 'Food')  # Default to Food
#                 expenses = Expense.objects.filter(user=user, category__name__icontains=category_name, date__month=now().month)
#                 total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You spent ₹{total} on {category_name} this month."
#             elif intent == "get_investments":
#                 logger.info(f"Handling get_investments for user {user.username}")
#                 investments = Investment.objects.filter(user=user)
#                 if investments:
#                     lines = [f"{i.investment_type.name}: ₹{i.amount} on {i.date}" for i in investments]
#                     response = "Your investments:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No investments recorded."
#             elif intent == "investment_get_profit":
#                 logger.info(f"Handling investment_get_profit for user {user.username}")
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     profit = Investment.objects.filter(user=user, investment_type__name=category_name).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your profit from {category_name} is ₹{profit}."
#                 else:
#                     total_profit = Investment.objects.filter(user=user).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your total profit from all investments is ₹{total_profit}."
#             elif intent == "investment_get_trend":
#                 logger.info(f"Handling investment_get_trend for user {user.username}")
#                 investments = Investment.objects.filter(user=user, date__gte=now().date() - timedelta(days=90))
#                 if investments:
#                     df = pd.DataFrame.from_records(investments.values('date', 'amount'))
#                     df = df.groupby('date')['amount'].sum().reset_index()
#                     trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                     response = f"Your average daily investment trend is ₹{trend:.2f}."
#                 else:
#                     response = "No investment data available for trend analysis."
#             # Add other intents from intent_mappings.json (to be verified)
#             else:
#                 logger.warning(f"Fallback triggered for intent: {intent}, query: {query}")
#                 response = "🤖 Sorry, I didn’t understand that. Can you rephrase your question?"

#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True

#         except Exception as e:
#             logger.error(f"Error in chatbot_view: {e}")
#             response = f"❌ Error: {e}"
#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True
#     print("Response: ",response)
#     return render(request, 'chatbot.html', {'response': response})



# import logging
# import re
# from datetime import datetime, timedelta
# from django.contrib.auth.decorators import login_required
# from django.shortcuts import render
# from django.utils.timezone import now
# from django.db.models import Sum
# from .models import Profile, Category, Budget, Bill, Expense, Income, IncomeCategory, Investment, InvestmentCategory
# import pandas as pd
# from .nlp_utils import detect_intent

# logger = logging.getLogger(__name__)

# @login_required
# def chatbot_view(request):
#     # Initialize session context
#     if 'chat_context' not in request.session:
#         request.session['chat_context'] = {
#             'history': [],  # (query, intent, response)
#             'current_intent': None,
#             'slots': {}
#         }
#     context = request.session['chat_context']

#     response = None
#     if request.method == 'POST':
#         query = request.POST.get('user_query', '').lower()
#         user = request.user
#         intent = detect_intent(query)
#         logger.info(f"Query: {query} | Intent: {intent}")
#         print(f"Query: {query} | Intent: {intent}")

#         try:
#             profile = Profile.objects.get(user=user)
#             context['history'].append((query, intent, None))
#             context['current_intent'] = intent

#             def extract_slots(text, intent_prefix):
#                 slots = {}
#                 # Extract amount (e.g., ₹500, 500, ₹500.00)
#                 amount_pattern = r"₹?\s*(\d+\.?\d*)"
#                 amount_match = re.search(amount_pattern, text)
#                 if amount_match:
#                     slots['amount'] = float(amount_match.group(1))

#                 # Extract category with improved keyword matching
#                 category_keywords = {
#                     'expense': ['food', 'dining', 'groceries', 'travel', 'electricity', 'utilities', 'subscriptions', 'meals'],
#                     'budget': ['food', 'dining', 'groceries', 'travel', 'subscriptions'],
#                     'bill': ['electricity', 'utilities', 'internet', 'phone', 'subscriptions'],
#                     'income': ['salary', 'freelance', 'bonus', 'business'],
#                     'investment': ['stocks', 'stock', 'mutual funds', 'bonds', 'business', 'equity']
#                 }
#                 text_words = text.lower().split()
#                 for keyword in category_keywords.get(intent_prefix, []):
#                     if keyword in text_words:
#                         if intent_prefix in ['expense', 'budget', 'bill']:
#                             category = Category.objects.filter(name__icontains=keyword).first()
#                         elif intent_prefix == 'income':
#                             category = IncomeCategory.objects.filter(name__icontains=keyword).first()
#                         elif intent_prefix == 'investment':
#                             category = InvestmentCategory.objects.filter(name__icontains=keyword).first()
#                         if category:
#                             slots['category'] = category.name
#                             break
#                 # Fallback to any word in query
#                 if 'category' not in slots:
#                     for word in text_words:
#                         word_cap = word.capitalize()
#                         if intent_prefix in ['expense', 'budget', 'bill']:
#                             category = Category.objects.filter(name__icontains=word_cap).first()
#                         elif intent_prefix == 'income':
#                             category = IncomeCategory.objects.filter(name__icontains=word_cap).first()
#                         elif intent_prefix == 'investment':
#                             category = InvestmentCategory.objects.filter(name__icontains=word_cap).first()
#                         if category:
#                             slots['category'] = category.name
#                             break

#                 # Extract date (default to current date for expenses)
#                 date_pattern = r"(next week|this month|last month|on (\w+ \d+)|(\d{4}-\d{2}-\d{2})|today|tomorrow)"
#                 date_match = re.search(date_pattern, text)
#                 if date_match:
#                     if date_match.group(1) == "next week":
#                         slots['date'] = (now().date() + timedelta(days=7)).strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "this month":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "last month":
#                         slots['date'] = (now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
#                     elif date_match.group(2):
#                         slots['date'] = pd.to_datetime(date_match.group(2), errors='coerce').strftime('%Y-%m-%d')
#                     elif date_match.group(3):
#                         slots['date'] = date_match.group(3)
#                     elif date_match.group(1) == "today":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "tomorrow":
#                         slots['date'] = (now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
#                 elif intent_prefix == 'expense':
#                     slots['date'] = now().date().strftime('%Y-%m-%d')  # Default to current date

#                 logger.info(f"Extracted slots for {intent_prefix}: {slots}")
#                 return slots

#             # Intent-based responses
#             if intent == "get_name":
#                 logger.info(f"Handling get_name for user {user.username}")
#                 response = f"Your name is {profile.name}."
#             elif intent == "get_total_spent":
#                 logger.info(f"Handling get_total_spent for user {user.username}")
#                 total = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You spent ₹{total} this month."
#             elif intent == "get_income_sources":
#                 logger.info(f"Handling get_income_sources for user {user.username}")
#                 categories = IncomeCategory.objects.all()
#                 response = "Income sources: " + ", ".join([c.name for c in categories])
#             elif intent == "get_food_expense":
#                 logger.info(f"Handling get_food_expense for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 # Check if query suggests adding an expense
#                 if any(word in query for word in ['add', 'spent', 'record', 'expense']):
#                     if 'amount' in context['slots'] and 'category' in context['slots']:
#                         amount = context['slots']['amount']
#                         category_name = context['slots']['category']
#                         try:
#                             category = Category.objects.get(name=category_name)
#                             Expense.objects.create(
#                                 user=user,
#                                 type='Cash',
#                                 category=category,
#                                 amount=amount,
#                                 date=now().date(),
#                                 notes=query
#                             )
#                             response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                             context['slots'] = {}
#                             context['current_intent'] = None
#                         except Category.DoesNotExist:
#                             response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
#                     else:
#                         response = "Please provide the amount and category (e.g., ₹500 for Food)."
#                 else:
#                     category_name = context['slots'].get('category', 'Food')  # Default to Food
#                     expenses = Expense.objects.filter(user=user, category__name__icontains=category_name, date__month=now().month)
#                     total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"You spent ₹{total} on {category_name} this month."
#             elif intent == "get_investments":
#                 logger.info(f"Handling get_investments for user {user.username}")
#                 investments = Investment.objects.filter(user=user)
#                 if investments:
#                     lines = [f"{i.investment_type.name}: ₹{i.amount} on {i.date}" for i in investments]
#                     response = "Your investments:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No investments recorded."
#             elif intent == "investment_get_profit":
#                 logger.info(f"Handling investment_get_profit for user {user.username}")
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 logger.info(f"investment_get_profit slots: {context['slots']}")
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     profit = Investment.objects.filter(user=user, investment_type__name=category_name).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your profit from {category_name} is ₹{profit}."
#                 else:
#                     total_profit = Investment.objects.filter(user=user).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your total profit from all investments is ₹{total_profit}."
#             elif intent == "investment_get_trend":
#                 logger.info(f"Handling investment_get_trend for user {user.username}")
#                 investments = Investment.objects.filter(user=user, date__gte=now().date() - timedelta(days=90))
#                 if investments:
#                     df = pd.DataFrame.from_records(investments.values('date', 'amount'))
#                     df = df.groupby('date')['amount'].sum().reset_index()
#                     trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                     response = f"Your average daily investment trend is ₹{trend:.2f}."
#                 else:
#                     response = "No investment data available for trend analysis."
#             elif intent in ["add_expense", "expense_add"]:
#                 logger.info(f"Handling {intent} for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 logger.info(f"{intent} slots: {context['slots']}")
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = Category.objects.get(name=category_name)
#                         Expense.objects.create(
#                             user=user,
#                             type='Cash',
#                             category=category,
#                             amount=amount,
#                             date=now().date(),
#                             notes=query
#                         )
#                         response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except Category.DoesNotExist:
#                         response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
#                 else:
#                     response = "Please provide the amount and category (e.g., ₹500 for Food)."
#             else:
#                 logger.warning(f"Fallback triggered for intent: {intent}, query: {query}")
#                 response = "🤖 Sorry, I didn’t understand that. Can you rephrase your question?"

#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True

#         except Exception as e:
#             logger.error(f"Error in chatbot_view: {e}")
#             response = f"❌ Error: {e}"
#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True
#     print("Response: ",response)
#     return render(request, 'chatbot.html', {'response': response})



# import logging
# import re
# from datetime import datetime, timedelta
# from django.contrib.auth.decorators import login_required
# from django.shortcuts import render
# from django.utils.timezone import now
# from django.db.models import Sum
# from .models import Profile, Category, Budget, Bill, Expense, Income, IncomeCategory, Investment, InvestmentCategory
# import pandas as pd
# from .nlp_utils import detect_intent

# logger = logging.getLogger(__name__)

# @login_required
# def chatbot_view(request):
#     # Initialize session context
#     if 'chat_context' not in request.session:
#         request.session['chat_context'] = {
#             'history': [],  # (query, intent, response)
#             'current_intent': None,
#             'slots': {}
#         }
#     context = request.session['chat_context']

#     response = None
#     if request.method == 'POST':
#         query = request.POST.get('user_query', '').lower()
#         user = request.user
#         intent = detect_intent(query)
#         logger.info(f"Query: {query} | Intent: {intent}")
#         print(f"Query: {query} | Intent: {intent}")

#         try:
#             profile = Profile.objects.get(user=user)
#             context['history'].append((query, intent, None))
#             context['current_intent'] = intent

#             def extract_slots(text, intent_prefix):
#                 slots = {}
#                 # Extract amount (e.g., ₹500, 500, ₹500.00, $5000)
#                 amount_pattern = r"[₹$]?\s*(\d+\.?\d*)"
#                 amount_match = re.search(amount_pattern, text)
#                 if amount_match:
#                     slots['amount'] = float(amount_match.group(1))

#                 # Extract category
#                 category_keywords = {
#                     'expense': ['food', 'dining', 'groceries', 'travel', 'electricity', 'utilities', 'subscriptions', 'meals'],
#                     'budget': ['food', 'dining', 'groceries', 'travel', 'subscriptions'],
#                     'bill': ['electricity', 'utilities', 'internet', 'phone', 'subscriptions'],
#                     'income': ['salary', 'freelance', 'bonus', 'business'],
#                     'investment': ['stocks', 'stock', 'mutual funds', 'bonds', 'business', 'equity']
#                 }
#                 text_words = text.lower().split()
#                 for keyword in category_keywords.get(intent_prefix, []):
#                     if keyword in text_words:
#                         if intent_prefix in ['expense', 'budget', 'bill']:
#                             category = Category.objects.filter(name__icontains=keyword).first()
#                         elif intent_prefix == 'income':
#                             category = IncomeCategory.objects.filter(name__icontains=keyword).first()
#                         elif intent_prefix == 'investment':
#                             category = InvestmentCategory.objects.filter(name__icontains=keyword).first()
#                         if category:
#                             slots['category'] = category.name
#                             break
#                 # Fallback to any word in query
#                 if 'category' not in slots:
#                     for word in text_words:
#                         word_cap = word.capitalize()
#                         if intent_prefix in ['expense', 'budget', 'bill']:
#                             category = Category.objects.filter(name__icontains=word_cap).first()
#                         elif intent_prefix == 'income':
#                             category = IncomeCategory.objects.filter(name__icontains=word_cap).first()
#                         elif intent_prefix == 'investment':
#                             category = InvestmentCategory.objects.filter(name__icontains=word_cap).first()
#                         if category:
#                             slots['category'] = category.name
#                             break

#                 # Extract date (default to current date for expenses)
#                 date_pattern = r"(next week|this month|last month|on (\w+ \d+)|(\d{4}-\d{2}-\d{2})|today|tomorrow)"
#                 date_match = re.search(date_pattern, text)
#                 if date_match:
#                     if date_match.group(1) == "next week":
#                         slots['date'] = (now().date() + timedelta(days=7)).strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "this month":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "last month":
#                         slots['date'] = (now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
#                     elif date_match.group(2):
#                         slots['date'] = pd.to_datetime(date_match.group(2), errors='coerce').strftime('%Y-%m-%d')
#                     elif date_match.group(3):
#                         slots['date'] = date_match.group(3)
#                     elif date_match.group(1) == "today":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "tomorrow":
#                         slots['date'] = (now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
#                 elif intent_prefix == 'expense':
#                     slots['date'] = now().date().strftime('%Y-%m-%d')

#                 logger.info(f"Extracted slots for {intent_prefix}: {slots}")
#                 return slots

#             # Intent-based responses
#             if intent == "get_name":
#                 logger.info(f"Handling get_name for user {user.username}")
#                 response = f"Your name is {profile.name}."
#             elif intent == "get_total_spent":
#                 logger.info(f"Handling get_total_spent for user {user.username}")
#                 total = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You spent ₹{total} this month."
#             elif intent == "get_income_sources":
#                 logger.info(f"Handling get_income_sources for user {user.username}")
#                 categories = IncomeCategory.objects.all()
#                 response = "Income sources: " + ", ".join([c.name for c in categories])
#             elif intent == "get_food_expense":
#                 logger.info(f"Handling get_food_expense for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 # Check for expense addition keywords
#                 if any(word in query for word in ['add', 'spent', 'record', 'expense']):
#                     if 'amount' in context['slots'] and 'category' in context['slots']:
#                         amount = context['slots']['amount']
#                         category_name = context['slots']['category']
#                         try:
#                             category = Category.objects.get(name=category_name)
#                             Expense.objects.create(
#                                 user=user,
#                                 type='Cash',
#                                 category=category,
#                                 amount=amount,
#                                 date=now().date(),
#                                 notes=query
#                             )
#                             response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                             context['slots'] = {}
#                             context['current_intent'] = None
#                         except Category.DoesNotExist:
#                             response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
#                     else:
#                         response = "Please provide the amount and category (e.g., ₹500 for Food)."
#                 else:
#                     category_name = context['slots'].get('category', 'Food')
#                     expenses = Expense.objects.filter(user=user, category__name__icontains=category_name, date__month=now().month)
#                     total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"You spent ₹{total} on {category_name} this month."
#             elif intent == "get_investments":
#                 logger.info(f"Handling get_investments for user {user.username}")
#                 investments = Investment.objects.filter(user=user)
#                 if investments:
#                     lines = [f"{i.investment_type.name}: ₹{i.amount} on {i.date}" for i in investments]
#                     response = "Your investments:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No investments recorded."
#             elif intent == "investment_get_profit":
#                 logger.info(f"Handling investment_get_profit for user {user.username}")
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 logger.info(f"investment_get_profit slots: {context['slots']}")
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     profit = Investment.objects.filter(user=user, investment_type__name=category_name).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your profit from {category_name} is ₹{profit}."
#                 else:
#                     total_profit = Investment.objects.filter(user=user).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your total profit from all investments is ₹{total_profit}."
#             elif intent == "investment_get_trend":
#                 logger.info(f"Handling investment_get_trend for user {user.username}")
#                 investments = Investment.objects.filter(user=user, date__gte=now().date() - timedelta(days=90))
#                 if investments:
#                     df = pd.DataFrame.from_records(investments.values('date', 'amount'))
#                     df = df.groupby('date')['amount'].sum().reset_index()
#                     trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                     response = f"Your average daily investment trend is ₹{trend:.2f}."
#                 else:
#                     response = "No investment data available for trend analysis."
#             elif intent in ["add_expense", "expense_add"]:
#                 logger.info(f"Handling {intent} for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 logger.info(f"{intent} slots: {context['slots']}")
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = Category.objects.get(name=category_name)
#                         Expense.objects.create(
#                             user=user,
#                             type='Cash',
#                             category=category,
#                             amount=amount,
#                             date=now().date(),
#                             notes=query
#                         )
#                         response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except Category.DoesNotExist:
#                         response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
#                 else:
#                     response = "Please provide the amount and category (e.g., ₹500 for Food)."
#             elif intent == "profile_get_savings":
#                 logger.info(f"Handling profile_get_savings for user {user.username}")
#                 response = f"Your current savings amount is ₹{profile.savings}."
#             elif intent == "get_total_income":
#                 logger.info(f"Handling get_total_income for user {user.username}")
#                 total = Income.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You earned ₹{total} this month."
#             elif intent == "profile_update_income":
#                 logger.info(f"Handling profile_update_income for user {user.username}")
#                 new_slots = extract_slots(query, 'income')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots']:
#                     amount = context['slots']['amount']
#                     profile.income = amount
#                     profile.save()
#                     response = f"Your monthly income has been updated to ₹{amount}."
#                 else:
#                     response = "Please provide the income amount (e.g., $5000)."
#             elif intent == "budget_get":
#                 logger.info(f"Handling budget_get for user {user.username}")
#                 new_slots = extract_slots(query, 'budget')
#                 context['slots'].update(new_slots)
#                 category_name = context['slots'].get('category', 'Groceries')
#                 budget = Budget.objects.filter(user=user, category__name__icontains=category_name).first()
#                 if budget:
#                     response = f"Your budget for {category_name} this month is ₹{budget.amount}."
#                 else:
#                     response = f"No budget set for {category_name}."
#             elif intent == "budget_get_all":
#                 logger.info(f"Handling budget_get_all for user {user.username}")
#                 budgets = Budget.objects.filter(user=user)
#                 if budgets:
#                     lines = [f"{b.category.name}: ₹{b.amount}" for b in budgets]
#                     response = "Your budget categories:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No budgets set."
#             elif intent == "category_get_list":
#                 logger.info(f"Handling category_get_list for user {user.username}")
#                 expenses = Expense.objects.filter(user=user, date__month=now().month).values('category__name').annotate(total=Sum('amount')).order_by('-total')
#                 if expenses:
#                     lines = [f"{e['category__name']}: ₹{e['total']}" for e in expenses]
#                     response = f"Your spending by category this month:<br>{lines[0]} (highest)"
#                 else:
#                     response = "No expenses recorded this month."
#             elif intent == "get_weekly_expenses":
#                 logger.info(f"Handling get_weekly_expenses for user {user.username}")
#                 last_month = now().replace(day=1) - timedelta(days=1)
#                 total = Expense.objects.filter(user=user, date__month=last_month.month, date__year=last_month.year).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your expenses last month were ₹{total}."
#             else:
#                 logger.warning(f"Fallback triggered for intent: {intent}, query: {query}")
#                 response = "🤖 Sorry, I didn’t understand that. Can you rephrase your question?"

#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True

#         except Exception as e:
#             logger.error(f"Error in chatbot_view: {e}")
#             response = f"❌ Error: {e}"
#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True
#     print("Response: ",response)
#     return render(request, 'chatbot.html', {'response': response})



import logging
import re
from datetime import datetime, timedelta
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils.timezone import now
from django.db.models import Sum
from .models import Profile, Category, Budget, Bill, Expense, Income, IncomeCategory, Investment, InvestmentCategory
import pandas as pd
from .nlp_utils import detect_intent

logger = logging.getLogger(__name__)

# @login_required
# def chatbot_view(request):
#     # Initialize session context
#     if 'chat_context' not in request.session:
#         request.session['chat_context'] = {
#             'history': [],  # (query, intent, response)
#             'current_intent': None,
#             'slots': {}
#         }
#     context = request.session['chat_context']

#     response = None
#     if request.method == 'POST':
#         query = request.POST.get('user_query', '').lower()
#         user = request.user
#         intent = detect_intent(query)
#         logger.info(f"Query: {query} | Intent: {intent}")
#         print(f"Query: {query} | Intent: {intent}")

#         try:
#             profile = Profile.objects.get(user=user)
#             context['history'].append((query, intent, None))
#             context['current_intent'] = intent

#             def extract_slots(text, intent_prefix):
#                 slots = {}
#                 # Extract amount (e.g., ₹500, 500, ₹500.00, $5000)
#                 amount_pattern = r"[₹$]?\s*(\d+\.?\d*)"
#                 amount_match = re.search(amount_pattern, text)
#                 if amount_match:
#                     slots['amount'] = float(amount_match.group(1))

#                 # Extract category
#                 category_keywords = {
#                     'expense': ['food', 'dining', 'groceries', 'travel', 'electricity', 'utilities', 'subscriptions', 'meals', 'entertainment', 'transportation'],
#                     'budget': ['food', 'dining', 'groceries', 'travel', 'subscriptions', 'entertainment', 'transportation'],
#                     'bill': ['electricity', 'utilities', 'internet', 'phone', 'subscriptions'],
#                     'income': ['salary', 'freelance', 'bonus', 'business'],
#                     'investment': ['stocks', 'stock', 'mutual funds', 'bonds', 'business', 'equity']
#                 }
#                 text_words = text.lower().split()
#                 for keyword in category_keywords.get(intent_prefix, []):
#                     if keyword in text_words:
#                         if intent_prefix in ['expense', 'budget', 'bill']:
#                             category = Category.objects.filter(name__icontains=keyword).first()
#                         elif intent_prefix == 'income':
#                             category = IncomeCategory.objects.filter(name__icontains=keyword).first()
#                         elif intent_prefix == 'investment':
#                             category = InvestmentCategory.objects.filter(name__icontains=keyword).first()
#                         if category:
#                             slots['category'] = category.name
#                             break
#                 # Fallback to any word in query
#                 if 'category' not in slots:
#                     for word in text_words:
#                         word_cap = word.capitalize()
#                         if intent_prefix in ['expense', 'budget', 'bill']:
#                             category = Category.objects.filter(name__icontains=word_cap).first()
#                         elif intent_prefix == 'income':
#                             category = IncomeCategory.objects.filter(name__icontains=word_cap).first()
#                         elif intent_prefix == 'investment':
#                             category = InvestmentCategory.objects.filter(name__icontains=word_cap).first()
#                         if category:
#                             slots['category'] = category.name
#                             break

#                 # Extract date or period
#                 date_pattern = r"(next week|this month|last month|past (\d+) months|on (\w+ \d+)|(\d{4}-\d{2}-\d{2})|today|tomorrow)"
#                 date_match = re.search(date_pattern, text)
#                 if date_match:
#                     if date_match.group(1) == "next week":
#                         slots['date'] = (now().date() + timedelta(days=7)).strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "this month":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "last month":
#                         slots['date'] = (now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
#                     elif date_match.group(2):  # past X months
#                         slots['months'] = int(date_match.group(2))
#                     elif date_match.group(3):
#                         slots['date'] = pd.to_datetime(date_match.group(3), errors='coerce').strftime('%Y-%m-%d')
#                     elif date_match.group(4):
#                         slots['date'] = date_match.group(4)
#                     elif date_match.group(1) == "today":
#                         slots['date'] = now().date().strftime('%Y-%m-%d')
#                     elif date_match.group(1) == "tomorrow":
#                         slots['date'] = (now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
#                 elif intent_prefix in ['expense', 'income', 'investment']:
#                     slots['date'] = now().date().strftime('%Y-%m-%d')

#                 logger.info(f"Extracted slots for {intent_prefix}: {slots}")
#                 return slots

#             # Intent-based responses
#             if intent == "get_name":
#                 logger.info(f"Handling get_name for user {user.username}")
#                 response = f"Your name is {profile.name}."
#             elif intent == "get_total_spent":
#                 logger.info(f"Handling get_total_spent for user {user.username}")
#                 total = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You spent ₹{total} this month."
#             elif intent == "get_income_sources":
#                 logger.info(f"Handling get_income_sources for user {user.username}")
#                 categories = IncomeCategory.objects.all()
#                 response = "Income sources: " + ", ".join([c.name for c in categories])
#             elif intent == "get_food_expense":
#                 logger.info(f"Handling get_food_expense for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 if any(word in query for word in ['add', 'spent', 'record', 'expense']):
#                     if 'amount' in context['slots'] and 'category' in context['slots']:
#                         amount = context['slots']['amount']
#                         category_name = context['slots']['category']
#                         try:
#                             category = Category.objects.get(name=category_name)
#                             Expense.objects.create(
#                                 user=user,
#                                 type='Cash',
#                                 category=category,
#                                 amount=amount,
#                                 date=now().date(),
#                                 notes=query
#                             )
#                             response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                             context['slots'] = {}
#                             context['current_intent'] = None
#                         except Category.DoesNotExist:
#                             response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
#                     else:
#                         response = "Please provide the amount and category (e.g., ₹500 for Food)."
#                 else:
#                     category_name = context['slots'].get('category', 'Food')
#                     expenses = Expense.objects.filter(user=user, category__name__icontains=category_name, date__month=now().month)
#                     total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"You spent ₹{total} on {category_name} this month."
#             elif intent == "get_investments":
#                 logger.info(f"Handling get_investments for user {user.username}")
#                 investments = Investment.objects.filter(user=user)
#                 if investments:
#                     lines = [f"{i.investment_type.name}: ₹{i.amount} on {i.date}" for i in investments]
#                     response = "Your investments:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No investments recorded."
#             elif intent == "investment_get_profit":
#                 logger.info(f"Handling investment_get_profit for user {user.username}")
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 logger.info(f"investment_get_profit slots: {context['slots']}")
#                 if 'category' in context['slots']:
#                     category_name = context['slots']['category']
#                     profit = Investment.objects.filter(user=user, investment_type__name=category_name).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your profit from {category_name} is ₹{profit}."
#                 else:
#                     total_profit = Investment.objects.filter(user=user).aggregate(Sum('profit'))['profit__sum'] or 0
#                     response = f"Your total profit from all investments is ₹{total_profit}."
#             elif intent == "investment_get_trend":
#                 logger.info(f"Handling investment_get_trend for user {user.username}")
#                 investments = Investment.objects.filter(user=user, date__gte=now().date() - timedelta(days=90))
#                 if investments:
#                     df = pd.DataFrame.from_records(investments.values('date', 'amount'))
#                     df = df.groupby('date')['amount'].sum().reset_index()
#                     trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
#                     response = f"Your average daily investment trend is ₹{trend:.2f}."
#                 else:
#                     response = "No investment data available for trend analysis."
#             elif intent in ["add_expense", "expense_add"]:
#                 logger.info(f"Handling {intent} for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 logger.info(f"{intent} slots: {context['slots']}")
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = Category.objects.get(name=category_name)
#                         Expense.objects.create(
#                             user=user,
#                             type='Cash',
#                             category=category,
#                             amount=amount,
#                             date=now().date(),
#                             notes=query
#                         )
#                         response = f"Expense of ₹{amount} for {category_name} added successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except Category.DoesNotExist:
#                         response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
#                 else:
#                     response = "Please provide the amount and category (e.g., ₹500 for Food)."
#             elif intent == "profile_get_savings":
#                 logger.info(f"Handling profile_get_savings for user {user.username}")
#                 response = f"Your current savings amount is ₹{profile.savings}."
#             elif intent == "get_total_income":
#                 logger.info(f"Handling get_total_income for user {user.username}")
#                 total = Income.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"You earned ₹{total} this month."
#             elif intent == "profile_update_income":
#                 logger.info(f"Handling profile_update_income for user {user.username}")
#                 new_slots = extract_slots(query, 'income')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots']:
#                     amount = context['slots']['amount']
#                     profile.income = amount
#                     profile.save()
#                     response = f"Your monthly income has been updated to ₹{amount}."
#                 else:
#                     response = "Please provide the income amount (e.g., $5000)."
#             elif intent == "budget_get":
#                 logger.info(f"Handling budget_get for user {user.username}")
#                 new_slots = extract_slots(query, 'budget')
#                 context['slots'].update(new_slots)
#                 category_name = context['slots'].get('category', 'Groceries')
#                 budget = Budget.objects.filter(user=user, category__name__icontains=category_name).first()
#                 if budget:
#                     response = f"Your budget for {category_name} this month is ₹{budget.amount}."
#                 else:
#                     response = f"No budget set for {category_name}."
#             elif intent == "budget_get_all":
#                 logger.info(f"Handling budget_get_all for user {user.username}")
#                 budgets = Budget.objects.filter(user=user)
#                 if budgets:
#                     lines = [f"{b.category.name}: ₹{b.amount}" for b in budgets]
#                     response = "Your budget categories:<br>" + "<br>".join(lines)
#                 else:
#                     response = "No budgets set."
#             elif intent == "category_get_list":
#                 logger.info(f"Handling category_get_list for user {user.username}")
#                 expenses = Expense.objects.filter(user=user, date__month=now().month).values('category__name').annotate(total=Sum('amount')).order_by('-total')
#                 if expenses:
#                     lines = [f"{e['category__name']}: ₹{e['total']}" for e in expenses]
#                     response = f"Your spending by category this month:<br>{lines[0]} (highest)"
#                 else:
#                     response = "No expenses recorded this month."
#             elif intent == "expense_get_by_date":
#                 logger.info(f"Handling expense_get_by_date for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 last_month = now().replace(day=1) - timedelta(days=1)
#                 expenses = Expense.objects.filter(user=user, date__month=last_month.month, date__year=last_month.year)
#                 total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your expenses last month were ₹{total}."
#             elif intent == "get_weekly_expenses":
#                 logger.info(f"Handling get_weekly_expenses for user {user.username}")
#                 new_slots = extract_slots(query, 'expense')
#                 context['slots'].update(new_slots)
#                 if 'category' in context['slots'] and 'months' in context['slots']:
#                     category_name = context['slots']['category']
#                     months = context['slots']['months']
#                     start_date = now().date() - timedelta(days=30 * months)
#                     expenses = Expense.objects.filter(
#                         user=user,
#                         category__name__icontains=category_name,
#                         date__gte=start_date
#                     )
#                     total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
#                     response = f"Your {category_name} expenses for the past {months} months were ₹{total}."
#                 else:
#                     response = "Please specify a category and time period (e.g., dining expenses for the past 3 months)."
#             elif intent == "investment_get_total":
#                 logger.info(f"Handling investment_get_total for user {user.username}")
#                 total = Investment.objects.filter(user=user).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"Your total investment portfolio value is ₹{total}."
#             elif intent == "get_profit_loss":
#                 logger.info(f"Handling get_profit_loss for user {user.username}")
#                 investments = Investment.objects.filter(user=user)
#                 if investments:
#                     lines = [f"{i.investment_type.name}: ₹{i.profit}" for i in investments if i.profit > 0]
#                     if lines:
#                         response = "Your profitable investments:<br>" + "<br>".join(lines)
#                     else:
#                         response = "No investments are currently profitable."
#                 else:
#                     response = "No investments recorded."
#             elif intent == "investment_add":
#                 logger.info(f"Handling investment_add for user {user.username}")
#                 new_slots = extract_slots(query, 'investment')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = InvestmentCategory.objects.get(name=category_name)
#                         Investment.objects.create(
#                             user=user,
#                             investment_type=category,
#                             amount=amount,
#                             date=now().date(),
#                             profit=0,
#                             notes=query
#                         )
#                         response = f"Investment of ₹{amount} in {category_name} added successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except InvestmentCategory.DoesNotExist:
#                         response = f"Investment category '{category_name}' not found. Available categories: {', '.join([c.name for c in InvestmentCategory.objects.all()])}."
#                 else:
#                     response = "Please provide the amount and investment category (e.g., $5000 for Stocks)."
#             elif intent == "get_income_savings":
#                 logger.info(f"Handling get_income_savings for user {user.username}")
#                 income = Income.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 expenses = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
#                 response = f"This month: Income ₹{income}, Expenses ₹{expenses}, Net ₹{income - expenses}."
#             elif intent in ["income_add", "income_category_add"]:
#                 logger.info(f"Handling {intent} for user {user.username}")
#                 new_slots = extract_slots(query, 'income')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = IncomeCategory.objects.get(name=category_name)
#                         Income.objects.create(
#                             user=user,
#                             category=category,
#                             amount=amount,
#                             date=now().date(),
#                             notes=query
#                         )
#                         response = f"Income of ₹{amount} for {category_name} added successfully!"
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except IncomeCategory.DoesNotExist:
#                         response = f"Income category '{category_name}' not found. Available categories: {', '.join([c.name for c in IncomeCategory.objects.all()])}."
#                 else:
#                     response = "Please provide the amount and income category (e.g., $1200 for Freelance)."
#             elif intent == "budget_update":
#                 logger.info(f"Handling budget_update for user {user.username}")
#                 new_slots = extract_slots(query, 'budget')
#                 context['slots'].update(new_slots)
#                 if 'amount' in context['slots'] and 'category' in context['slots']:
#                     amount = context['slots']['amount']
#                     category_name = context['slots']['category']
#                     try:
#                         category = Category.objects.get(name=category_name)
#                         budget = Budget.objects.filter(user=user, category=category).first()
#                         if budget:
#                             budget.amount += amount
#                             budget.save()
#                             response = f"Budget for {category_name} increased by ₹{amount} to ₹{budget.amount}."
#                         else:
#                             Budget.objects.create(
#                                 user=user,
#                                 category=category,
#                                 amount=amount,
#                                 date=now().date()
#                             )
#                             response = f"New budget of ₹{amount} for {category_name} created."
#                         context['slots'] = {}
#                         context['current_intent'] = None
#                     except Category.DoesNotExist:
#                         response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
#                 else:
#                     response = "Please provide the amount and category (e.g., $50 for Transportation)."
#             else:
#                 logger.warning(f"Fallback triggered for intent: {intent}, query: {query}")
#                 response = "🤖 Sorry, I didn’t understand that. Can you rephrase your question?"

#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True

#         except Exception as e:
#             logger.error(f"Error in chatbot_view: {e}")
#             response = f"❌ Error: {e}"
#             context['history'][-1] = (query, intent, response)
#             request.session.modified = True
#     print("Response: ",response)
#     return render(request, 'chatbot.html', {'response': response})



import logging
import re
from datetime import datetime, timedelta
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils.timezone import now
from django.db.models import Sum
from .models import Profile, Category, Budget, Bill, Expense, Income, IncomeCategory, Investment, InvestmentCategory
import pandas as pd
from .nlp_utils import detect_intent

logger = logging.getLogger(__name__)

@login_required
def chatbot_view(request):
    # Initialize session context
    if 'chat_context' not in request.session:
        request.session['chat_context'] = {
            'history': [],  # (query, intent, response)
            'current_intent': None,
            'slots': {}
        }
    context = request.session['chat_context']

    response = None
    if request.method == 'POST':
        query = request.POST.get('user_query', '').lower()
        user = request.user
        intent = detect_intent(query)
        logger.info(f"Query: {query} | Intent: {intent}")
        print(f"Query: {query} | Intent: {intent}")

        try:
            profile = Profile.objects.get(user=user)
            context['history'].append((query, intent, None))
            context['current_intent'] = intent

            def extract_slots(text, intent_prefix):
                slots = {}
                # Extract amount (e.g., ₹500, 500, ₹500.00, $5000)
                amount_pattern = r"[₹$]?\s*(\d+\.?\d*)"
                amount_match = re.search(amount_pattern, text)
                if amount_match:
                    slots['amount'] = float(amount_match.group(1))

                # Extract category
                category_keywords = {
                    'expense': ['food', 'dining', 'groceries', 'travel', 'electricity', 'utilities', 'subscriptions', 'meals', 'entertainment', 'transportation'],
                    'budget': ['food', 'dining', 'groceries', 'travel', 'subscriptions', 'entertainment', 'transportation'],
                    'bill': ['electricity', 'utilities', 'internet', 'phone', 'subscriptions'],
                    'income': ['salary', 'freelance', 'bonus', 'business'],
                    'investment': ['stocks', 'stock', 'mutual funds', 'bonds', 'business', 'equity', 'cryptocurrencies']
                }
                text_words = text.lower().split()
                for keyword in category_keywords.get(intent_prefix, []):
                    if keyword in text_words:
                        if intent_prefix in ['expense', 'budget', 'bill']:
                            category = Category.objects.filter(name__icontains=keyword).first()
                        elif intent_prefix == 'income':
                            category = IncomeCategory.objects.filter(name__icontains=keyword).first()
                        elif intent_prefix == 'investment':
                            category = InvestmentCategory.objects.filter(name__icontains=keyword).first()
                        if category:
                            slots['category'] = category.name
                            break
                # Fallback to any word in query
                if 'category' not in slots:
                    for word in text_words:
                        word_cap = word.capitalize()
                        if intent_prefix in ['expense', 'budget', 'bill']:
                            category = Category.objects.filter(name__icontains=word_cap).first()
                        elif intent_prefix == 'income':
                            category = IncomeCategory.objects.filter(name__icontains=word_cap).first()
                        elif intent_prefix == 'investment':
                            category = InvestmentCategory.objects.filter(name__icontains=word_cap).first()
                        if category:
                            slots['category'] = category.name
                            break

                # Extract date or period
                date_pattern = r"(next week|this month|last month|past (\d+) months|on (\w+ \d+)|(\d{4}-\d{2}-\d{2})|today|tomorrow)"
                date_match = re.search(date_pattern, text)
                if date_match:
                    if date_match.group(1) == "next week":
                        slots['date'] = (now().date() + timedelta(days=7)).strftime('%Y-%m-%d')
                    elif date_match.group(1) == "this month":
                        slots['date'] = now().date().strftime('%Y-%m-%d')
                    elif date_match.group(1) == "last month":
                        slots['date'] = (now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
                    elif date_match.group(2):  # past X months
                        slots['months'] = int(date_match.group(2))
                    elif date_match.group(3):
                        slots['date'] = pd.to_datetime(date_match.group(3), errors='coerce').strftime('%Y-%m-%d')
                    elif date_match.group(4):
                        slots['date'] = date_match.group(4)
                    elif date_match.group(1) == "today":
                        slots['date'] = now().date().strftime('%Y-%m-%d')
                    elif date_match.group(1) == "tomorrow":
                        slots['date'] = (now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
                elif intent_prefix in ['expense', 'income', 'investment']:
                    slots['date'] = now().date().strftime('%Y-%m-%d')

                logger.info(f"Extracted slots for {intent_prefix}: {slots}")
                return slots

            # Intent-based responses
            if intent == "get_name":
                logger.info(f"Handling get_name for user {user.username}")
                response = f"Your name is {profile.name}."
            elif intent == "get_total_spent":
                logger.info(f"Handling get_total_spent for user {user.username}")
                total = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
                response = f"You spent ₹{total} this month."
            elif intent == "get_income_sources":
                logger.info(f"Handling get_income_sources for user {user.username}")
                categories = IncomeCategory.objects.all()
                response = "Income sources: " + ", ".join([c.name for c in categories])
            elif intent == "get_food_expense":
                logger.info(f"Handling get_food_expense for user {user.username}")
                new_slots = extract_slots(query, 'expense')
                context['slots'].update(new_slots)
                if any(word in query for word in ['add', 'spent', 'record']) and 'amount' in context['slots']:
                    if 'category' in context['slots']:
                        amount = context['slots']['amount']
                        category_name = context['slots']['category']
                        try:
                            category = Category.objects.get(name=category_name)
                            Expense.objects.create(
                                user=user,
                                type='Cash',
                                category=category,
                                amount=amount,
                                date=now().date(),
                                notes=query
                            )
                            response = f"Expense of ₹{amount} for {category_name} added successfully!"
                            context['slots'] = {}
                            context['current_intent'] = None
                        except Category.DoesNotExist:
                            response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
                    else:
                        response = "Please provide the amount and category (e.g., ₹500 for Food)."
                else:
                    category_name = context['slots'].get('category', 'Food')
                    expenses = Expense.objects.filter(user=user, category__name__icontains=category_name, date__month=now().month)
                    total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
                    response = f"You spent ₹{total} on {category_name} this month."
            elif intent == "get_investments":
                logger.info(f"Handling get_investments for user {user.username}")
                investments = Investment.objects.filter(user=user)
                if investments:
                    lines = [f"{i.investment_type.name}: ₹{i.amount} on {i.date}" for i in investments]
                    response = "Your investments:<br>" + "<br>".join(lines)
                else:
                    response = "No investments recorded."
            elif intent == "investment_get_profit":
                logger.info(f"Handling investment_get_profit for user {user.username}")
                new_slots = extract_slots(query, 'investment')
                context['slots'].update(new_slots)
                logger.info(f"investment_get_profit slots: {context['slots']}")
                if 'category' in context['slots']:
                    category_name = context['slots']['category']
                    profit = Investment.objects.filter(user=user, investment_type__name=category_name).aggregate(Sum('profit'))['profit__sum'] or 0
                    response = f"Your profit from {category_name} is ₹{profit}."
                else:
                    total_profit = Investment.objects.filter(user=user).aggregate(Sum('profit'))['profit__sum'] or 0
                    response = f"Your total profit from all investments is ₹{total_profit}."
            elif intent == "investment_get_trend":
                logger.info(f"Handling investment_get_trend for user {user.username}")
                investments = Investment.objects.filter(user=user, date__gte=now().date() - timedelta(days=90))
                if investments:
                    df = pd.DataFrame.from_records(investments.values('date', 'amount'))
                    df = df.groupby('date')['amount'].sum().reset_index()
                    trend = df['amount'].rolling(window=7).mean().iloc[-1] if len(df) > 7 else df['amount'].mean()
                    response = f"Your average daily investment trend is ₹{trend:.2f}."
                else:
                    response = "No investment data available for trend analysis."
            elif intent in ["add_expense", "expense_add"]:
                logger.info(f"Handling {intent} for user {user.username}")
                new_slots = extract_slots(query, 'expense')
                context['slots'].update(new_slots)
                logger.info(f"{intent} slots: {context['slots']}")
                if 'amount' in context['slots'] and 'category' in context['slots']:
                    amount = context['slots']['amount']
                    category_name = context['slots']['category']
                    try:
                        category = Category.objects.get(name=category_name)
                        Expense.objects.create(
                            user=user,
                            type='Cash',
                            category=category,
                            amount=amount,
                            date=now().date(),
                            notes=query
                        )
                        response = f"Expense of ₹{amount} for {category_name} added successfully!"
                        context['slots'] = {}
                        context['current_intent'] = None
                    except Category.DoesNotExist:
                        response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
                else:
                    response = "Please provide the amount and category (e.g., ₹500 for Food)."
            elif intent == "profile_get_savings":
                logger.info(f"Handling profile_get_savings for user {user.username}")
                response = f"Your current savings amount is ₹{profile.savings}."
            elif intent == "get_total_income":
                logger.info(f"Handling get_total_income for user {user.username}")
                total = Income.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
                response = f"You earned ₹{total} this month."
            elif intent == "profile_update_income":
                logger.info(f"Handling profile_update_income for user {user.username}")
                new_slots = extract_slots(query, 'income')
                context['slots'].update(new_slots)
                if 'amount' in context['slots']:
                    amount = context['slots']['amount']
                    profile.income = amount
                    profile.save()
                    response = f"Your monthly income has been updated to ₹{amount}."
                else:
                    response = "Please provide the income amount (e.g., $5000)."
            elif intent == "budget_get":
                logger.info(f"Handling budget_get for user {user.username}")
                new_slots = extract_slots(query, 'budget')
                context['slots'].update(new_slots)
                category_name = context['slots'].get('category', 'Groceries')
                budget = Budget.objects.filter(user=user, category__name__icontains=category_name).first()
                if budget:
                    response = f"Your budget for {category_name} this month is ₹{budget.amount}."
                else:
                    response = f"No budget set for {category_name}."
            elif intent == "budget_get_all":
                logger.info(f"Handling budget_get_all for user {user.username}")
                budgets = Budget.objects.filter(user=user)
                if budgets:
                    lines = [f"{b.category.name}: ₹{b.amount}" for b in budgets]
                    response = "Your budget categories:<br>" + "<br>".join(lines)
                else:
                    response = "No budgets set."
            elif intent == "category_get_list":
                logger.info(f"Handling category_get_list for user {user.username}")
                expenses = Expense.objects.filter(user=user, date__month=now().month).values('category__name').annotate(total=Sum('amount')).order_by('-total')
                if expenses:
                    lines = [f"{e['category__name']}: ₹{e['total']}" for e in expenses]
                    response = f"Your spending by category this month:<br>{lines[0]} (highest)"
                else:
                    response = "No expenses recorded this month."
            elif intent == "expense_get_by_date":
                logger.info(f"Handling expense_get_by_date for user {user.username}")
                new_slots = extract_slots(query, 'expense')
                context['slots'].update(new_slots)
                last_month = now().replace(day=1) - timedelta(days=1)
                expenses = Expense.objects.filter(user=user, date__month=last_month.month, date__year=last_month.year)
                total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
                response = f"Your expenses last month were ₹{total}."
            elif intent == "expense_get_by_category_date":
                logger.info(f"Handling expense_get_by_category_date for user {user.username}")
                new_slots = extract_slots(query, 'expense')
                context['slots'].update(new_slots)
                if 'category' in context['slots'] and 'months' in context['slots']:
                    category_name = context['slots']['category']
                    months = context['slots']['months']
                    start_date = now().date() - timedelta(days=30 * months)
                    expenses = Expense.objects.filter(
                        user=user,
                        category__name__icontains=category_name,
                        date__gte=start_date
                    )
                    total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
                    response = f"Your {category_name} expenses for the past {months} months were ₹{total}."
                else:
                    response = "Please specify a category and time period (e.g., dining expenses for the past 3 months)."
            elif intent == "get_weekly_expenses":
                logger.info(f"Handling get_weekly_expenses for user {user.username}")
                new_slots = extract_slots(query, 'expense')
                context['slots'].update(new_slots)
                if 'category' in context['slots'] and 'months' in context['slots']:
                    category_name = context['slots']['category']
                    months = context['slots']['months']
                    start_date = now().date() - timedelta(days=30 * months)
                    expenses = Expense.objects.filter(
                        user=user,
                        category__name__icontains=category_name,
                        date__gte=start_date
                    )
                    total = expenses.aggregate(Sum('amount'))['amount__sum'] or 0
                    response = f"Your {category_name} expenses for the past {months} months were ₹{total}."
                else:
                    response = "Please specify a category and time period (e.g., dining expenses for the past 3 months)."
            elif intent == "investment_get_total":
                logger.info(f"Handling investment_get_total for user {user.username}")
                total = Investment.objects.filter(user=user).aggregate(Sum('amount'))['amount__sum'] or 0
                response = f"Your total investment portfolio value is ₹{total}."
            elif intent == "get_profit_loss":
                logger.info(f"Handling get_profit_loss for user {user.username}")
                investments = Investment.objects.filter(user=user)
                if investments:
                    lines = [f"{i.investment_type.name}: ₹{i.profit}" for i in investments if i.profit > 0]
                    if lines:
                        response = "Your profitable investments:<br>" + "<br>".join(lines)
                    else:
                        response = "No investments are currently profitable."
                else:
                    response = "No investments recorded."
            elif intent == "investment_add":
                logger.info(f"Handling investment_add for user {user.username}")
                new_slots = extract_slots(query, 'investment')
                context['slots'].update(new_slots)
                if 'amount' in context['slots'] and 'category' in context['slots']:
                    amount = context['slots']['amount']
                    category_name = context['slots']['category']
                    try:
                        category = InvestmentCategory.objects.get(name=category_name)
                        Investment.objects.create(
                            user=user,
                            investment_type=category,
                            amount=amount,
                            date=now().date(),
                            profit=0,
                            notes=query
                        )
                        response = f"Investment of ₹{amount} in {category_name} added successfully!"
                        context['slots'] = {}
                        context['current_intent'] = None
                    except InvestmentCategory.DoesNotExist:
                        response = f"Investment category '{category_name}' not found. Available categories: {', '.join([c.name for c in InvestmentCategory.objects.all()])}."
                else:
                    response = "Please provide the amount and investment category (e.g., $5000 for Stocks)."
            elif intent == "get_income_savings":
                logger.info(f"Handling get_income_savings for user {user.username}")
                income = Income.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
                expenses = Expense.objects.filter(user=user, date__month=now().month).aggregate(Sum('amount'))['amount__sum'] or 0
                response = f"This month: Income ₹{income}, Expenses ₹{expenses}, Net ₹{income - expenses}."
            elif intent in ["income_add", "income_category_add"]:
                logger.info(f"Handling {intent} for user {user.username}")
                new_slots = extract_slots(query, 'income')
                context['slots'].update(new_slots)
                if 'amount' in context['slots'] and 'category' in context['slots']:
                    amount = context['slots']['amount']
                    category_name = context['slots']['category']
                    try:
                        category = IncomeCategory.objects.get(name=category_name)
                        Income.objects.create(
                            user=user,
                            category=category,
                            amount=amount,
                            date=now().date(),
                            notes=query
                        )
                        response = f"Income of ₹{amount} for {category_name} added successfully!"
                        context['slots'] = {}
                        context['current_intent'] = None
                    except IncomeCategory.DoesNotExist:
                        response = f"Income category '{category_name}' not found. Available categories: {', '.join([c.name for c in IncomeCategory.objects.all()])}."
                else:
                    response = "Please provide the amount and income category (e.g., $1200 for Freelance)."
            elif intent == "budget_update":
                logger.info(f"Handling budget_update for user {user.username}")
                new_slots = extract_slots(query, 'budget')
                context['slots'].update(new_slots)
                if 'amount' in context['slots'] and 'category' in context['slots']:
                    amount = context['slots']['amount']
                    category_name = context['slots']['category']
                    try:
                        category = Category.objects.get(name=category_name)
                        budget = Budget.objects.filter(user=user, category=category).first()
                        if budget:
                            budget.amount += amount
                            budget.save()
                            response = f"Budget for {category_name} increased by ₹{amount} to ₹{budget.amount}."
                        else:
                            Budget.objects.create(
                                user=user,
                                category=category,
                                amount=amount
                            )
                            response = f"New budget of ₹{amount} for {category_name} created."
                        context['slots'] = {}
                        context['current_intent'] = None
                    except Category.DoesNotExist:
                        response = f"Category '{category_name}' not found. Available categories: {', '.join([c.name for c in Category.objects.all()])}."
                else:
                    response = "Please provide the amount and category (e.g., $50 for Transportation)."
            else:
                logger.warning(f"Fallback triggered for intent: {intent}, query: {query}")
                response = "🤖 Sorry, I didn’t understand that. Can you rephrase your question?"

            context['history'][-1] = (query, intent, response)
            request.session.modified = True

        except Exception as e:
            logger.error(f"Error in chatbot_view: {e}")
            response = f"❌ Error: {e}"
            context['history'][-1] = (query, intent, response)
            request.session.modified = True
    print("Response: ",response)
    return render(request, 'chatbot.html', {'response': response})
