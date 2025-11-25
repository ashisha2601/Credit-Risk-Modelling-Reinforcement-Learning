# Quick Extraction Helper - Fill This Out

Use this template to extract values from RBI reports and Census data.

## 📊 RBI Financial Stability Report (Latest: ______)

### Default Rates
- Overall retail default rate: ______%
- Personal loans default rate: ______%
- Credit cards default rate: ______%
- Home loans default rate: ______%
- Microfinance default rate: ______%

**Source Table**: _________________
**Page Number**: _________________

### Interest Rates
- Personal loans:
  - Minimum: ______%
  - Maximum: ______%
  - Average: ______%
- Credit cards:
  - Minimum: ______%
  - Maximum: ______%
  - Average: ______%
- Home loans:
  - Minimum: ______%
  - Maximum: ______%
  - Average: ______%

**Source Table**: _________________
**Page Number**: _________________

### Loan Characteristics
- Average loan-to-income ratio: ______x
- Typical loan tenure (months): ______
- Average personal loan amount: ₹______

**Source Table**: _________________
**Page Number**: _________________

---

## 📈 RBI Trends and Progress (Year: ______)

### Credit Growth
- Personal loans growth rate: ______%
- Credit card growth rate: ______%
- Overall retail credit growth: ______%

**Source Table**: _________________
**Page Number**: _________________

### Portfolio Statistics
- Average loan amount: ₹______
- Median loan amount: ₹______
- Loan size distribution:
  - < ₹1L: ______%
  - ₹1L-5L: ______%
  - ₹5L-10L: ______%
  - > ₹10L: ______%

**Source Table**: _________________
**Page Number**: _________________

---

## 🏘️ Census 2011

### Demographics
- Average age: ______ years
- Age distribution:
  - 18-25: ______%
  - 26-35: ______%
  - 36-50: ______%
  - 51-70: ______%

- Urban population: ______%
- Rural population: ______%

**Source Table**: _________________

### Education
- High School: ______%
- Graduate: ______%
- Post-Graduate: ______%
- Professional: ______%

**Source Table**: _________________

### Occupation
- fmt: ______%
- Self-employed: ______%
- Business: ______%
- Agriculture: ______%
- Others: ______%

**Source Table**: _________________

---

## 💰 Income Data (MOSPI/PLFS/Census)

### Urban Income (Monthly)
- Mean: ₹______
- Median: ₹______
- Standard Deviation: ₹______
- Distribution shape: _______________

**Source**: _________________
**Year**: _________________

### Rural Income (Monthly)
- Mean: ₹______
- Median: ₹______
- Standard Deviation: ₹______

**Source**: _________________
**Year**: _________________

### Income by Occupation
- Salaried mean: ₹______
- Self-employed mean: ₹______
- Business mean: ₹______

**Source**: _________________

---

## 🏦 Banking Access

### Account Penetration
- Bank account coverage: ______%
- Credit card penetration: ______%
- Average accounts per person: ______

**Source**: _________________

### Digital Adoption
- Digital banking usage: ______%
- UPI adoption: ______%
- Mobile banking: ______%

**Source**: _________________

---

## 📝 Update config/priors_template.yaml

After filling this out, update the values in `config/priors_template.yaml`:

```yaml
default_rates:
  overall_retail: 0.025  # ← Fill from above
  personal_loans: 0.03   # ← Fill from above
  # ... etc

interest_rates:
  personal_loans:
    min: 0.10  # ← Fill from above
    max: 0.24  # ← Fill from above
    # ... etc
```

---

## 🔍 Where to Find Each Value

| Value | Source | Where to Look |
|-------|--------|---------------|
| Default rates | RBI FSR | Table: "Banking System Performance" or "Sectoral NPAs" |
| Interest rates | RBI Bulletin | Table: "Lending Rates" or "Bank Credit" |
| Loan amounts | RBI Trends | Table: "Personal Loans Outstanding" |
| Demographics | Census 2011 | "Primary Census Abstract" Excel |
| Income | MOSPI/PLFS | "Household Consumption" reports |
| Education | Census 2011 | "Educational Level" tables |
| Occupation | Census 2011 | "Workers by Category" tables |

---

## ✅ Checklist

- [ ] Downloaded latest RBI FSR
- [ ] Downloaded latest Trends and Progress
- [ ] Downloaded Census 2011 tables
- [ ] Extracted all default rates
- [ ] Extracted all interest rates
- [ ] Extracted income distributions
- [ ] Extracted demographics
- [ ] Updated config/priors_template.yaml
- [ ] Verified values are reasonable
- [ ] Documented sources in data card

---

**Time Estimate**: 2-3 hours for complete extraction

**Tip**: Don't aim for perfection! Get reasonable estimates for Review-1, refine later.


