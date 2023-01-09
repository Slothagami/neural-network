
// Script to download transcripts from microsoft stream vods
// Using OBS to record audio

var seen_text = []
var transcript = document.querySelector("#transcript-content > search-event > div > div.content > div.transclude-content > ng-transclude > transcript > div > div.transcript-inner.new > virtual-list > div.virtual-list > div.scrollable-content")
var string = ""

// scroll down transcript to load it all into memory
var interval = setInterval(() => {
    for(let el of transcript.querySelectorAll("li")) {
        if(!seen_text.includes(el.innerText)) {
            string += "\n" + el.innerText
            seen_text.push(el.innerText)
        }
    }
}, 100);

// add button to copy the text
let btn = document.createElement("button")
btn.onclick = () => {
    clearInterval(interval)
    navigator.clipboard.writeText(string)
    btn.parentElement.removeChild(btn)
}

btn.innerText = "Copy Transcript"
document.body.appendChild(btn)

btn.style.position = "fixed"
btn.style.bottom   = "10px"
btn.style.right    = "10px"
btn.style.color    = "white"
btn.style.backgroundColor = "black"
