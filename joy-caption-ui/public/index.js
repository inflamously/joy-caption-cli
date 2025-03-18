const fields = {
    form: null,
    path: null,
}

const app = {
    registerEvents: () => {
        fields.form.addEventListener('submit', (ev) => {
            ev.preventDefault()
            console.log(ev);
            return false;
        })

        fields.path.addEventListener('change', (ev) => {
            console.log(ev);
            // fetch(`http://localhost:5678/api/v1/paths?file=${}`).then((response) => {
            //     console.log(response)
            // })
        })
    }
}


function main() {
    fields.form = document.getElementById('joy-caption-form')
    fields.path = document.getElementById('path')

    app.registerEvents()
}

main()