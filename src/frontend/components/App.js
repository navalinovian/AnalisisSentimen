import React, { Component } from 'react';
import { render } from "react-dom";
import DjangoCSRFToken from 'django-react-csrftoken'

export default class App extends Component {
    constructor(props){
        super(props);
    }
    render() {
        return (
            <div>
                <h1>Testing React Code</h1>
                <form action="/" method="POST" enctype="multipart/form-data">
                    <DjangoCSRFToken/>
                    {/* <input type="text" name="lala"/> */}
                    <input type="file" name="filename" id="filename"/>
                    <input type="submit" name="submit"/>
                </form>
                <div>
                    {dataJson}
                </div>
                <div>
                </div>
            </div>
        )
    }
}

const appDiv = document.getElementById("App");
render(<App/>, appDiv);