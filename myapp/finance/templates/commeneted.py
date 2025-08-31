# def dashboard(request):
#     profile = Profile.objects.get(user=request.user)




#     budget_qs = Budget.objects.filter(user=request.user)
#     bills_qs = Bill.objects.filter(user=request.user)
#     expenses_qs = Expense.objects.filter(user=request.user)

#     # Convert QuerySets to DataFrames
#     budget_df = read_frame(budget_qs, fieldnames=['category_id', 'amount'])
#     bills_df = read_frame(bills_qs, fieldnames=['category_id', 'amount'])
#     expenses_df = read_frame(expenses_qs, fieldnames=['category_id', 'amount'])

#     # Rename amount columns
#     budget_df.rename(columns={"amount": "Budgeted"}, inplace=True)
#     bills_df.rename(columns={"amount": "Spent"}, inplace=True)
#     expenses_df.rename(columns={"amount": "Spent"}, inplace=True)

#     # Combine bills and expenses spending
#     spending_df = pd.concat([bills_df, expenses_df], ignore_index=True).groupby("category_id", as_index=False).sum()

#     # Merge budget and spending data
#     df = pd.merge(budget_df, spending_df, on="category_id", how="outer").fillna(0)

#     # Convert category_id to string for better visualization
#     df["Category"] = df["category_id"].str.extract(r'\((\d+)\)').astype(int)
#     # category = [Category.objects.get(id=id) for id in df["Category"]]
#     # Get all category IDs from df
#     category = [Category.objects.filter(id=id).first().name if Category.objects.filter(id=id).exists() else "Unknown" for id in df["Category"]]


#     for c in category:
#          print(c)

#     df["Category"] = category

#     print(df.head())  # Debugging step
#     # Ensure numeric types for plotting
#     df["Budgeted"] = pd.to_numeric(df["Budgeted"], errors='coerce')
#     df["Spent"] = pd.to_numeric(df["Spent"], errors='coerce')


#     # Create the bar chart
#     fig = px.bar(df, x="Category", y=["Budgeted", "Spent"],
#                  title="Spending vs Budget",
#                  barmode="group",
#                  labels={"value": "Amount", "variable": "Type"},
#                  color_discrete_map={"Budgeted": "blue", "Spent": "red"})

#     chart_html = opy.plot(fig, auto_open=False, output_type="div")
#     # plt.figure(figsize=(8,4))
#     # plt.bar(df['Budgeted'], df['Spent'])
#     # plt.xlabel('Budget')
#     # plt.ylabel('amount')
#     # buffer = BytesIO()
#     # plt.save(buffer, format='png')
#     # chart_html =base64.b64decode(buffer.getvalue()).decode()
#     # buffer.close

#     return render(request, 'dashboard.html', {'profile':profile, 'chart':chart_html})

# def parse_expense_text(text, user):
#     # Extract amount using regex (₹ followed by digits)
#     amount_pattern = r"₹?(\d+)"
#     amount_match = re.search(amount_pattern, text)
#     amount = int(amount_match.group(1)) if amount_match else None

#     # Extract category by checking the text for predefined categories
#     category = Category.get_category(text)

#     # Default to 'Cash' for the type
#     expense_type = 'Cash'

#     # Use the entire text as notes
#     notes = text

#     if amount and category:
#         # Create Expense entry
#         expense = Expense(
#             user=user,
#             type=expense_type,
#             category=category,
#             amount=amount,
#             date=date.today(),  # Current date
#             notes=notes
#         )
#         expense.save()
#         return expense
#     else:
#         return None  # If parsing failed