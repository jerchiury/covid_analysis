library(httr)
library(jsonlite)
setwd('C:\\Users\\Jerry\\Desktop\\Jerry\\projects\\covid19')
today=Sys.Date()
cases.zip='https://www150.statcan.gc.ca/n1/pub/13-26-0003/2020001/COVID19-eng.zip' # individual leevel data source

get.data=function(url, prefix, query='', write.to.file=F, file.prefix=today){ #provincial or health region level data
  query=paste(query, collapse='&')
  url=paste0(url, '?', query)
  print(url)
  temp=GET(url)
  temp=rawToChar(temp$content)
  a<<-temp
  temp=fromJSON(temp)
  names=names(temp)
  for(n in names){
    var.name=paste0(prefix, '.', n)
    eval.text=paste0(var.name, '<<-temp$', n)
    eval(parse(text=eval.text))
    if(write.to.file){
      file.name=paste0(file.prefix, '_', gsub('\\.', '_', prefix), '_', n, '.csv')
      eval.text=paste0('write.csv(', var.name, ',"', file.name, '", row.names=F)')
      eval(parse(text=eval.text))
    }
  }
}

get.ind.data=function(url, write.to.file=F){ # individual level data
  temp <- tempfile()
  download.file('https://www150.statcan.gc.ca/n1/pub/13-26-0003/2020001/COVID19-eng.zip',temp)
  names=unzip(temp, list=T)$Name
  for(n in names){
    var.name=tolower(gsub('-eng.csv', '', n))
    assign(var.name, read.csv(unz(temp, n)), envir=globalenv())
    if(write.to.file){
      eval.text=paste0('write.csv(', var.name, ',"', n, '", row.names=F)')
      eval(parse(text=eval.text))
      }
  }
  unlink(temp)
  }

get.data('https://api.opencovid.ca/timeseries', 'ts.hr', 'loc=hr')
get.data('https://api.opencovid.ca/timeseries', 'ts.prov', 'loc=prov')

get.ind.data(case.zip)

