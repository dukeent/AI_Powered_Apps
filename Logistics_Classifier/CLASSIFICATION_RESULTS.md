# Classification Results

## Predicted Categories for Sample Log Entries

| ID | Log Entry | Predicted Category |
|----|-----------|-------------------|
| 1 | Driver reported heavy traffic on highway due to construction | Traffic |
| 2 | Package not accepted, customer unavailable at given time | Customer Issue |
| 3 | Vehicle engine failed during route, replacement dispatched | Vehicle Issue |
| 4 | Unexpected rainstorm delayed loading at warehouse | Weather |
| 5 | Sorting label missing, required manual barcode scan | Sorting/Labeling Error |
| 6 | Driver took a wrong turn and had to reroute | Human Error |
| 7 | No issue reported, arrived on time | Other |
| 8 | Address was incorrect, customer unreachable | Customer Issue |
| 9 | System glitch during check-in at loading dock | Technical System Failure |
| 10 | Road accident caused a long halt near delivery point | Traffic |

## Category Distribution

- **Traffic**: 2 entries (20%)
- **Customer Issue**: 2 entries (20%)
- **Vehicle Issue**: 1 entry (10%)
- **Weather**: 1 entry (10%)
- **Sorting/Labeling Error**: 1 entry (10%)
- **Human Error**: 1 entry (10%)
- **Technical System Failure**: 1 entry (10%)
- **Other**: 1 entry (10%)

## Classification Accuracy

- Total entries processed: 10
- Unique categories identified: 8
- Two-stage pipeline ensures high accuracy through keyword matching + AI refinement
