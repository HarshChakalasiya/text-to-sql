# Invoices in Supply Chain Finance

An invoice is a commercial document issued by a seller (supplier) to a buyer stating amounts due for goods or services. In this SCF platform, each invoice links one buyer and one seller, has an amount and currency, issue date, due date, and a lifecycle status.

Statuses:
- open: raised by the seller, awaiting buyer approval
- approved: buyer has confirmed the invoice is valid for payment or financing
- financed: early payment has been disbursed against the invoice under a financing program
- paid: the buyer (or obligor) has settled the full invoice amount
- overdue: payment is past due_date and not fully settled
- disputed: buyer or seller has raised a commercial dispute

Analysts often filter overdue invoices by amount thresholds, buyer, or seller to monitor credit exposure.