from django.db import models
from multiselectfield import MultiSelectField
from django.utils.text import slugify
from django.contrib.auth.models import User

# Create your models here.


class Profile(models.Model):
    # sources = models.MultiSelectField()
    name = models.CharField(max_length=200)
    img_url = models.ImageField( upload_to="profile_images/", blank=True, null=True)
  
    income = models.IntegerField()
    savings = models.IntegerField()
    slug = models.SlugField(unique=True, max_length=255)
    user = models.ForeignKey(User, on_delete=models.CASCADE , null=True)

    
    def save(self, *args, **kwargs):
        self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    @property
    def formatted_img_url(self):
        url = self.img_url if self.img_url.__str__().startswith(("http://","https://")) else self.img_url.url
        return url

    def __str__(self):
        return self.name
    


# class Category(models.Model):
#     name = models.CharField(max_length=200, unique=True)

class Category(models.Model):
    name = models.CharField(max_length=200, unique=True)

    

    @staticmethod
    def get_category(name):
        categories = [
            "Food", "Entertainment", "Rent", "Groceries", "Utilities",
            "Transportation", "Healthcare", "Education", "Insurance",
            "Gifts & Donations", "Dining Out", "Subscriptions",
            "Home Maintenance", "Personal Care", "Childcare", "Debt Payments",
            "Cloths & Accessories", "Travel", "Pets", "Shopping"
        ]
        for category_name in categories:
            if category_name.lower() in name.lower():
                return Category.objects.get_or_create(name=category_name)[0]
        return None


class Budget(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    amount = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)


from django.db import models
from django.contrib.auth.models import User

class Bill(models.Model):
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

    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Links bill to a user
    category = models.ForeignKey(Category, on_delete=models.CASCADE)

    title = models.CharField(max_length=255)  # Bill title (e.g., Electricity Bill)
    amount = models.DecimalField(max_digits=10, decimal_places=2)  # Bill amount
    due_date = models.DateField()  # Due date of the bill
    billing_cycle = models.CharField(max_length=20, choices=BILLING_CYCLE_CHOICES, default='monthly')  # Billing cycle
    custom_cycle_days = models.PositiveIntegerField(blank=True, null=True, help_text="Applicable only if 'Custom' cycle is selected.")  # Custom cycle input

    created_at = models.DateTimeField(auto_now_add=True)  # Auto timestamp when bill is created
    updated_at = models.DateTimeField(auto_now=True)  # Auto timestamp when bill is updated
    is_paid = models.BooleanField(default=False)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-due_date']  # Orders bills by closest due date






class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    type =models.CharField(max_length=200)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    amount = models.IntegerField()
    date = models.DateField()
    notes = models.TextField(blank=True)




class IncomeCategory(models.Model):
    name = models.CharField(max_length=200, unique=True)


class Income(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    category = models.ForeignKey(IncomeCategory, on_delete=models.CASCADE)
    amount = models.IntegerField()
    date = models.DateField()
    notes = models.TextField(blank=True)

class InvestmentCategory(models.Model):
    name= models.CharField(max_length=200, unique= True)
    def __str__(self):
        return self.name



class Investment(models.Model):
    user = models.ForeignKey(User,  on_delete=models.CASCADE, null=True)
    investment_type = models.ForeignKey(InvestmentCategory, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField()
    profit = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    loss = models.DecimalField(max_digits=10, decimal_places=2, default=0)


