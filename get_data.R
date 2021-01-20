library(httr)
library(jsonlite)
setwd('C:\\Users\\Jerry\\Desktop\\Jerry\\projects\\covid19')
today=Sys.Date()

get.data=function(url, prefix, query='', write.to.file=F, file.prefix=today){
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
      file.name=paste0(file.prefix, '_', prefix, '_', n, '.csv')
      eval.text=paste0('write.csv(', var.name, ',"', file.name, '", row.names=F)')
      eval(parse(text=eval.text))
      }
    }
  }
get.data('https://api.opencovid.ca/timeseries', 'ts', write.to.file=T)

