from finance.models import InvestmentCategory
from django.core.management.base import BaseCommand




class Command(BaseCommand):
    help = " inserts categories"


    
    def handle(self, *args, **kwargs):
        InvestmentCategory.objects.all().delete()
        categories =[
                "Stock", "Mutual Funds", "Fixed Deposits", "Real Estate"
                , "Cryptocurrencies", "Bussiness"
            ]


        for category in categories:
            InvestmentCategory.objects.create(name=category)

        self.stdout.write(self.style.SUCCESS("INSERTION COMPLETED!"))

