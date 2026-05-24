import type { DocumentType } from "@/lib/types";

export interface SampleDoc {
  id: string;
  label: string;
  document_type: DocumentType;
  difficulty: "easy" | "medium" | "hard";
  content: string;
}

export const SAMPLES: SampleDoc[] = [
  {
    id: "invoice_easy",
    label: "Invoice · easy (clean)",
    document_type: "invoice",
    difficulty: "easy",
    content: `INVOICE

Invoice number: INV-2026-0042
Issue date: 2026-03-15

From: Acme Corp
  100 Main St, San Francisco, CA 94103

To: Beta Industries Inc.
  500 Market St, San Francisco, CA 94105

Description: Consulting services Q1 2026
Subtotal: $4,500.00
Tax (8.5%): $382.50
Total: $4,882.50

Payment due: 2026-04-15
`,
  },
  {
    id: "invoice_medium",
    label: "Invoice · medium (prose + long date)",
    document_type: "invoice",
    difficulty: "medium",
    content: `Globex Corporation
----------------
Invoice: GLX-2026-554
Issued on: March 22nd, 2026

Professional services:
  Architecture review                $3,500
  Implementation support              $7,200

Total due: $10,700.00
Net 30
`,
  },
  {
    id: "invoice_hard",
    label: "Invoice · hard (Spanish, decimal comma)",
    document_type: "invoice",
    difficulty: "hard",
    content: `FACTURA

Número de factura: FAC-2026-0214
Fecha de emisión: 11/06/2026

Emisor: Aceitunas García S.L.
  Calle Mayor 12, 28013 Madrid

Cliente: Distribuidora Norte

Concepto: Suministro mensual aceitunas premium
Base imponible: 2.450,00 €
IVA (21%):       514,50 €
TOTAL:         2.964,50 €
`,
  },
  {
    id: "receipt_easy",
    label: "Receipt · easy",
    document_type: "receipt",
    difficulty: "easy",
    content: `*** Whole Foods Market ***
2210 Bay St, San Francisco
Receipt #88421
2026-03-12 14:33

Organic bananas (1.2 lb)   $1.98
Almond milk 32oz           $4.49
Sourdough loaf             $5.99
--------------------------------
Subtotal                  $12.46
Tax                        $1.06
TOTAL                     $13.52

VISA ****4221  APPROVED
`,
  },
  {
    id: "contract_easy",
    label: "Contract · easy (NDA)",
    document_type: "contract",
    difficulty: "easy",
    content: `MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement (the 'Agreement') is entered into as of January 15, 2026 (the 'Effective Date') by and between:

  Party A: Acme Corp, a Delaware corporation
  Party B: Beta Industries Inc., a California corporation

Term: This Agreement shall remain in effect for a period of two (2) years from the Effective Date.

Governing Law: This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, USA.

[signatures...]
`,
  },
];
