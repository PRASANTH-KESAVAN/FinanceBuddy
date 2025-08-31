from django.core.management.base import BaseCommand
from django.utils.timezone import now
from django.core.mail import send_mail
from finance.models import Bill
from datetime import timedelta

class Command(BaseCommand):
    help = 'Send email reminders for upcoming bills'

    def handle(self, *args, **kwargs):
        upcoming = now().date() + timedelta(days=3)
        bills = Bill.objects.filter(due_date=upcoming, is_paid=False)

        for bill in bills:
            user = bill.user
            subject = f"Reminder: {bill.name} is due in 3 days"
            message = f"Hi {user.username},\n\nThis is a reminder that your bill \"{bill.name}\" of ₹{bill.amount} is due on {bill.due_date}.\n\nPlease make sure to pay it on time.\n\nRegards,\nSmart Budgeting App"
            send_mail(subject, message, 'noreply@smartbudget.com', [user.email])
            self.stdout.write(f"Sent reminder to {user.email} for {bill.name}")
