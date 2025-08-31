from finance.models import Category
from django.core.management.base import BaseCommand




class Command(BaseCommand):
    help = " inserts categories"


    
    def handle(self, *args, **kwargs):
        Category.objects.all().delete()
        categories =[
                "Food", "Entertainment", "Rent", "Groceries", "Utilities",
                "Transportation", "Healthcare", "Education", "Insurance",
                "Gifts & Donations",  "Dining Out", "Subscriptions",
                "Home Maintenance", "Personal Care", "Childcare", "Debt Payments","Cloths & Accessories"
            ]


        for category in categories:
            Category.objects.create(name=category)

        self.stdout.write(self.style.SUCCESS("INSERTION COMPLETED!"))

