export namespace joycaption {
	
	export class FileBuffer {
	    bytes: number[];
	    name: string;
	
	    static createFrom(source: any = {}) {
	        return new FileBuffer(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.bytes = source["bytes"];
	        this.name = source["name"];
	    }
	}

}

