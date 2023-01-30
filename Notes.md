# Measuring Mainstream

### Journals
- [x] American Economic Review  
- [x] Econometrica
- [x] International Economic Review
- [ ] Journal of Economic Theory (missing form constellate)
- [x] Journal of Political Economy
- [x] Quarterly Journal of Economics 
- [x] Review of Economic Studies
- [x] Review of Economics and Statistics
- [x] Economica
- [x] Economic Journal
- [x] Bell Journal of Economics (only 70s-80s)

Downloaded 19/09/2022

Constallate ids:
* 31bcd322-032f-2ba5-37c4-c2ca96952ebf (Blue Ribbons - 1900-1954)
* 1a0cd895-717b-dcc2-940d-a4d7ed069aea (Blue Ribbons - 1955-1984)
* 03aa84ad-ec5e-290c-69e7-721535714c9e (Blue Ribbons - 1985-2022)

Downloaded 08/10/2022

Constallate ids:
* 4788f182-8ec3-8eb8-ac5b-a1b318b03dbb (Economica and Economic Journal)
* c55cf30b-737b-9a83-3325-8d82a71bd8e6 (Bell Journal of Economics)

### Keys 
* pagination
* provider
* publisher
* identifier
* outputFormat
* keyphrase
* title
* creator
* volumeNumber
* language
* pageStart
* sequence
* tdmCategory
* unigramCount
* datePublished
* pageCount
* docType
* publicationYear
* isPartOf
* id
* doi
* wordCount
* pageEnd
* docSubType
* bigramCount
* url
* trigramCount
* sourceCategory
* issueNumber

### DATASET METADATA (From old work)

| Field           | Type      | Description                              |
| --------------- | --------- | ---------------------------------------- |
| creator         | List[str] | authors                                  |
| docType         | str       | categorical type                         |
| doi             | str       | unique digital identifier                |
| id              | str       | unique identifier (possible primary key) |
| isPartOf        | str       | journal title                            |
| language        | List[str] | languages (to filter)                    |
| outputFormat    | List[str] | data available (1-2-3grams etc) (filter) |
| provider        | str       | portico or JSTOR                         |
| publicationYear | int       | year                                     |
| tdmCategory     | List[str] | categories                               |
| sourceCategory  | List[str] | other categories                         |
| title           | str       | title                                    |

## Impact Factor
[https://github.com/clarivate/wosjournals-python-client]
