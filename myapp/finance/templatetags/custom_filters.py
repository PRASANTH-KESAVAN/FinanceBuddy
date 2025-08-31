from datetime import datetime
from django import template
from dateutil.relativedelta import relativedelta

register = template.Library()

increment = {
    'one_time': 0, 
    'daily': 1, 
    'weekly': 1,
    'bi_weekly': 2, 
    'monthly': 1, 
    'bi_monthly': 2, 
    'quarterly': 3, 
    'semi_annual': 6,
    'annual': 1 
}

@register.filter(name='date_format')
def date_format(bill):
    """
    Custom template filter to format the next billing date as "Renews on Month Day, Year".
    """
    # Ensure `bill.updated_at` is a valid datetime object
    date = bill.updated_at  
    if not isinstance(date, datetime):  
        try:
            date = datetime.strptime(date, '%Y-%m-%d')  # Adjust format if needed
        except (ValueError, TypeError):
            return "Invalid Date"

    increment_type = bill.billing_cycle  

    if increment_type in increment:
        new_date = date + relativedelta(months=increment[increment_type])
        return f"Renews on {new_date.strftime('%B %d, %Y')}"  # Example: "Renews on March 26, 2025"

    return "Invalid Cycle"
