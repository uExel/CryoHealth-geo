# ADR 0001 — Earth observation platform: Copernicus Data Space Ecosystem

## Context
PRD Open Question 2, blocking R2: production Sentinel-2 monitoring needs an EO platform.
Two named candidates — Google Earth Engine (GEE) and Copernicus Data Space Ecosystem
(CDSE) — plus a practical constraint on how this ADR gets written: an agent session has
no Google Cloud project, no GEE registration, and no CDSE account, and creating an
account or entering credentials on Shaan's behalf is out of scope for what an agent does
unilaterally. That constraint turned out to matter for the decision itself, not just for
how the accompanying proof-of-concept got built (see "Proof-of-concept" below).

## Comparison

| | Google Earth Engine | Copernicus Data Space Ecosystem |
|---|---|---|
| **Cost model for a funded commercial product** | Noncommercial tier explicitly excludes "fee-for-service" work for a paying funder — a GLOF early-warning platform built under a GLOF-II-adjacent funding relationship does not clearly qualify. Commercial pricing is not published; "contact Google or a certified partner" is the only path, meaning cost is unknown until a sales conversation happens. | Free tier is transparent and published: 10,000 openEO processing credits/month per account, documented quotas for STAC/download APIs, no commercial-use exclusion. Predictable to budget against without a sales cycle. |
| **API maturity for this use case** | Mature, huge community, the Python API is arguably the easiest path to write NDWI/cloud-mask code quickly — this is real and shouldn't be understated. | STAC + openEO are open standards (not Copernicus-specific), so the code isn't locked to one vendor's API shape; the same STAC-search-then-read pattern works against other STAC catalogs (this ADR's own proof-of-concept exploits exactly that portability, see below). |
| **Governance fit** | A single US company's commercial terms, changeable, historically has tightened (the 2024 commercial-tier introduction itself is an example of the terms shifting under existing users). | ESA/EU public infrastructure, Sentinel data's own government-funded distribution channel — no vendor discontinuation risk beyond the mission itself. |
| **Onboarding cost today** | OAuth + a Google Cloud project + Earth Engine registration approval (not instant). | Free self-service registration at dataspace.copernicus.eu, typically approved quickly, no project/billing setup required for the free tier. |

## Decision
**Copernicus Data Space Ecosystem**, via its openEO or STAC API. The deciding factor is
the cost model: GEE's noncommercial tier is a real legal risk for a company shipping a
funded product (explicitly excludes fee-for-service work), and its commercial tier has
no published price — that's not a foundation to build a scheduled production pipeline
on without first having a pricing conversation with Google. CDSE's free tier is
transparent, documented, and sized (10,000 credits/month) well beyond what monitoring
25 lakes on a multi-day revisit cadence needs. GEE's easier Python API is real but not
decisive against an open cost/legal risk.

**Action needed from Shaan, not done here:** a free CDSE account
(dataspace.copernicus.eu) and its OAuth2 client credentials, to be added to this
service's environment when the production pipeline (not this spike) is built.

## Proof-of-concept — what it actually validates, and the honest gap

The DoD asked for "fetch one Sentinel-2 scene for one lake AOI and compute NDWI water
extent on the chosen platform." That literally requires CDSE credentials this session
doesn't have and won't create. Rather than fabricate a result or block entirely, the
proof-of-concept validates the same NDWI/cloud-mask/area pipeline logic against **real,
current Sentinel-2 L2A imagery** pulled from **Microsoft Planetary Computer's STAC API**,
which is genuinely anonymous — no account, no key, confirmed via its own docs ("The STAC
API is public and can be accessed anonymously"). This proves the actual
computation — NDWI math, SCL-based cloud masking, water-pixel area estimate — against
real data today, honestly, without pretending to have CDSE access this session doesn't
have.

STAC is a standard; Planetary Computer and CDSE both expose Sentinel-2 L2A through
STAC-compliant catalogs with the same band-asset shape (`B03`, `B08`, `SCL`, etc.). The
pipeline's scene-source is behind a small interface (`pipeline/stac_source.py`) for
exactly this reason: swapping the catalog URL and adding CDSE's OAuth2 flow is the
remaining step, not a rewrite. That swap is intentionally **not** done in this task —
it needs the real account from the "action needed" line above.

**Run it:** `uv run python -m pipeline.poc --lake shishper`
