from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Buyer(Base):
    __tablename__ = "buyers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    industry: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(80), nullable=False)
    credit_rating: Mapped[str] = mapped_column(String(10), nullable=False)

    invoices: Mapped[list["Invoice"]] = relationship(back_populates="buyer")


class Seller(Base):
    __tablename__ = "sellers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    industry: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(80), nullable=False)

    invoices: Mapped[list["Invoice"]] = relationship(back_populates="seller")


class Program(Base):
    __tablename__ = "programs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    buyer_id: Mapped[int] = mapped_column(ForeignKey("buyers.id"), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    max_tenor_days: Mapped[int] = mapped_column(nullable=False, default=90)
    discount_rate_bps: Mapped[int] = mapped_column(nullable=False, default=250)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="active")

    buyer: Mapped[Buyer] = relationship()
    financings: Mapped[list["Financing"]] = relationship(back_populates="program")


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(primary_key=True)
    invoice_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    buyer_id: Mapped[int] = mapped_column(ForeignKey("buyers.id"), nullable=False)
    seller_id: Mapped[int] = mapped_column(ForeignKey("sellers.id"), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    issue_date: Mapped[date] = mapped_column(Date, nullable=False)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    # open | approved | financed | paid | overdue | disputed

    buyer: Mapped[Buyer] = relationship(back_populates="invoices")
    seller: Mapped[Seller] = relationship(back_populates="invoices")
    financings: Mapped[list["Financing"]] = relationship(back_populates="invoice")
    payments: Mapped[list["Payment"]] = relationship(back_populates="invoice")


class Financing(Base):
    __tablename__ = "financings"

    id: Mapped[int] = mapped_column(primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)
    program_id: Mapped[int] = mapped_column(ForeignKey("programs.id"), nullable=False)
    funded_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    discount_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    disbursement_date: Mapped[date] = mapped_column(Date, nullable=False)
    maturity_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    # disbursed | settled | defaulted

    invoice: Mapped[Invoice] = relationship(back_populates="financings")
    program: Mapped[Program] = relationship(back_populates="financings")


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    payment_date: Mapped[date] = mapped_column(Date, nullable=False)
    method: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    invoice: Mapped[Invoice] = relationship(back_populates="payments")


SCHEMA_DESCRIPTION = """
Tables and relationships for the Supply Chain Finance (SCF) database:

buyers(id, name, industry, country, credit_rating)
  - Corporate buyers that approve supplier invoices for financing.

sellers(id, name, industry, country)
  - Suppliers / vendors who raise invoices against buyers.

programs(id, name, buyer_id -> buyers.id, currency, max_tenor_days, discount_rate_bps, status)
  - Financing programs sponsored by a buyer. discount_rate_bps is basis points (250 = 2.50%).

invoices(id, invoice_number, buyer_id -> buyers.id, seller_id -> sellers.id, amount, currency,
         issue_date, due_date, status)
  - status values: open, approved, financed, paid, overdue, disputed
  - An invoice is overdue when status = 'overdue' OR (status in ('open','approved') AND due_date < CURRENT_DATE).

financings(id, invoice_id -> invoices.id, program_id -> programs.id, funded_amount, discount_amount,
           disbursement_date, maturity_date, status)
  - status values: disbursed, settled, defaulted
  - Invoice discounting / early payment against approved invoices.

payments(id, invoice_id -> invoices.id, amount, payment_date, method, created_at)
  - Buyer settlements against invoices.

Join hints:
- invoices.buyer_id = buyers.id
- invoices.seller_id = sellers.id
- programs.buyer_id = buyers.id
- financings.invoice_id = invoices.id
- financings.program_id = programs.id
- payments.invoice_id = invoices.id
""".strip()