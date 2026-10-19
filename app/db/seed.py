"""Seed synthetic Supply Chain Finance data."""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.models import Buyer, Financing, Invoice, Payment, Program, Seller

INDUSTRIES = ["Automotive", "Retail", "Manufacturing", "Pharma", "Electronics", "FMCG"]
COUNTRIES = ["India", "UAE", "Singapore", "USA", "Germany"]
CREDIT_RATINGS = ["AAA", "AA", "A", "BBB", "BB"]
PAYMENT_METHODS = ["NEFT", "RTGS", "ACH", "Wire"]


def _d(n: float) -> Decimal:
    return Decimal(str(round(n, 2)))


def seed_database(db: Session, *, seed: int = 42) -> dict[str, int]:
    rng = random.Random(seed)

    # Clear existing rows (child tables first)
    for model in (Payment, Financing, Invoice, Program, Seller, Buyer):
        db.query(model).delete()
    db.commit()

    buyers = [
        Buyer(
            name=name,
            industry=rng.choice(INDUSTRIES),
            country=rng.choice(COUNTRIES),
            credit_rating=rng.choice(CREDIT_RATINGS),
        )
        for name in [
            "Nova Motors Ltd",
            "Bright Retail Corp",
            "Helix Pharma Inc",
            "Orbit Electronics",
            "GreenLeaf FMCG",
            "Apex Manufacturing",
            "Summit Auto Group",
            "Pulse Consumer Goods",
        ]
    ]
    db.add_all(buyers)
    db.flush()

    sellers = [
        Seller(
            name=name,
            industry=rng.choice(INDUSTRIES),
            country=rng.choice(COUNTRIES),
        )
        for name in [
            "Precision Parts Co",
            "Metro Packaging",
            "ChemSource Labs",
            "Silicon Components",
            "FreshPack Foods",
            "SteelForge Works",
            "Delta Logistics",
            "Nova Textiles",
            "Rapid Moulds",
            "CloudWare IT Services",
            "Harbor Chemicals",
            "Vista Components",
        ]
    ]
    db.add_all(sellers)
    db.flush()

    programs: list[Program] = []
    for buyer in buyers:
        for i in range(rng.randint(1, 2)):
            programs.append(
                Program(
                    name=f"{buyer.name.split()[0]} SCF Program {i + 1}",
                    buyer_id=buyer.id,
                    currency="INR",
                    max_tenor_days=rng.choice([60, 90, 120]),
                    discount_rate_bps=rng.choice([180, 220, 250, 300, 350]),
                    status=rng.choice(["active", "active", "paused"]),
                )
            )
    db.add_all(programs)
    db.flush()

    today = date.today()
    invoices: list[Invoice] = []
    statuses = ["open", "approved", "financed", "paid", "overdue", "disputed"]
    status_weights = [15, 20, 25, 25, 12, 3]

    for i in range(220):
        buyer = rng.choice(buyers)
        seller = rng.choice(sellers)
        issue = today - timedelta(days=rng.randint(5, 180))
        due = issue + timedelta(days=rng.choice([30, 45, 60, 90]))
        status = rng.choices(statuses, weights=status_weights, k=1)[0]
        if status == "overdue":
            due = today - timedelta(days=rng.randint(1, 45))
        elif status == "paid":
            due = today - timedelta(days=rng.randint(10, 90))
        amount = _d(rng.uniform(25_000, 2_500_000))
        invoices.append(
            Invoice(
                invoice_number=f"INV-{10000 + i}",
                buyer_id=buyer.id,
                seller_id=seller.id,
                amount=amount,
                currency="INR",
                issue_date=issue,
                due_date=due,
                status=status,
            )
        )
    db.add_all(invoices)
    db.flush()

    financings: list[Financing] = []
    payments: list[Payment] = []
    buyer_programs = {b.id: [p for p in programs if p.buyer_id == b.id] for b in buyers}

    for inv in invoices:
        if inv.status in ("financed", "paid") and rng.random() < 0.85:
            prog_list = buyer_programs.get(inv.buyer_id) or programs
            program = rng.choice(prog_list)
            discount = _d(float(inv.amount) * program.discount_rate_bps / 10000)
            funded = _d(float(inv.amount) - float(discount))
            disbursement = inv.issue_date + timedelta(days=rng.randint(1, 15))
            maturity = disbursement + timedelta(days=program.max_tenor_days)
            fin_status = "settled" if inv.status == "paid" else "disbursed"
            if rng.random() < 0.03:
                fin_status = "defaulted"
            financings.append(
                Financing(
                    invoice_id=inv.id,
                    program_id=program.id,
                    funded_amount=funded,
                    discount_amount=discount,
                    disbursement_date=disbursement,
                    maturity_date=maturity,
                    status=fin_status,
                )
            )

        if inv.status == "paid":
            payments.append(
                Payment(
                    invoice_id=inv.id,
                    amount=inv.amount,
                    payment_date=inv.due_date + timedelta(days=rng.randint(-5, 20)),
                    method=rng.choice(PAYMENT_METHODS),
                )
            )
        elif inv.status == "financed" and rng.random() < 0.2:
            partial = _d(float(inv.amount) * rng.uniform(0.2, 0.5))
            payments.append(
                Payment(
                    invoice_id=inv.id,
                    amount=partial,
                    payment_date=today - timedelta(days=rng.randint(1, 30)),
                    method=rng.choice(PAYMENT_METHODS),
                )
            )

    db.add_all(financings)
    db.add_all(payments)
    db.commit()

    return {
        "buyers": len(buyers),
        "sellers": len(sellers),
        "programs": len(programs),
        "invoices": len(invoices),
        "financings": len(financings),
        "payments": len(payments),
    }