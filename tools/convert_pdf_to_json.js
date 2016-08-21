let fs = require('fs'),
        PDFParser = require("./node_modules/pdf2json/PDFParser");

    let pdfParser = new PDFParser();

    pdfParser.on("pdfParser_dataError", errData => console.error(errData.parserError) );
    pdfParser.on("pdfParser_dataReady", pdfData => {
        fs.writeFile("./EBLAmap1.json", JSON.stringify(pdfData));
    });

    pdfParser.loadPDF("./pdfs/EBLAmap1.pdf")
