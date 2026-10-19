# Invoice Discounting and Financing Flows

Invoice discounting (also called early payment or reverse factoring when buyer-sponsored) advances cash to the seller against an approved invoice. The financier funds a discounted amount before the invoice due date; the discount is the financing fee.

In this system:
- A buyer sponsors one or more financing programs with a max tenor (days) and discount_rate_bps
- discount_rate_bps is in basis points; 250 means 2.50%
- A financing record stores funded_amount, discount_amount, disbursement_date, maturity_date, and status
- Financing statuses: disbursed (cash paid to seller), settled (buyer repaid the financier), defaulted

Typical flow:
1. Seller raises invoice
2. Buyer approves invoice
3. Seller (or buyer) requests financing under a program
4. Financier disburses funded_amount to seller
5. Buyer pays at maturity / due date; financing is settled