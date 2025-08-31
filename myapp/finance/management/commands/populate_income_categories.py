from finance.models import IncomeCategory
from django.core.management.base import BaseCommand




class Command(BaseCommand):
    help = " inserts categories"


    
    def handle(self, *args, **kwargs):
        IncomeCategory.objects.all().delete()
        categories =[
                "Salary/Wages", "Business Income", "Freelance/Side Hustle", 
                "Rental Income", "Interest Income", "Pension", "Bonuses",
                
            ]


        for category in categories:
            IncomeCategory.objects.create(name=category)

        self.stdout.write(self.style.SUCCESS("INSERTION COMPLETED!"))

