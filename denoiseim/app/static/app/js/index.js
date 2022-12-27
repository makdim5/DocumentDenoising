function printDocument(documentId) {
    var divContents = document.getElementById(documentId).textContent;
    var a = window.open('', '', 'height=500, width=1000');
    a.document.write(divContents);
    a.document.close();
    a.print();
}

btn = document.getElementById("print_btn")
btn.onclick = () => {
    printDocument("mytext")
}