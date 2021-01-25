library(httr)
library(jsonlite)
setwd('C:\\Users\\Jerry\\Desktop\\Jerry\\projects\\covid19')
today=Sys.Date()
cases.zip='https://www150.statcan.gc.ca/n1/pub/13-26-0003/2020001/COVID19-eng.zip' # individual leevel data source
cases.github.url='https://github.com/ccodwg/Covid19Canada/raw/master/cases.csv'
github.age.case.map=c(url='https://github.com/ccodwg/Covid19Canada/raw/master/other/age_map_cases.csv', name='age_case_map')
github.age.mortality.map=c(url='https://github.com/ccodwg/Covid19Canada/raw/master/other/age_map_mortality.csv', name='age_mortality_map')
github.hr.map=c(url='https://github.com/ccodwg/Covid19Canada/raw/master/other/hr_map.csv', name='hr_map')
github.prov.map=c(url='https://github.com/ccodwg/Covid19Canada/raw/master/other/prov_map.csv', name='prov_map')

get.json.data=function(url, prefix, query='', write.to.file=F, file.prefix='updated'){ #provincial or health region level data
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
  download.file('https://www150.statcan.gc.ca/n1/pub/13-26-0003/2020001/COVID19-eng.zip', temp)
  names=unzip(temp, list=T)$Name[1]
  for(n in names){
    file.text=paste0(readLines(unz(temp, n)), 'AAAAAAAAAA')
    file.text=gsub(',AAAAAAAAAA','', file.text, fixed=T)
    file.text=gsub('AAAAAAAAAA','', file.text, fixed=T)
    conn=textConnection(file.text)
    
    var.name=tolower(gsub('-eng.csv', '', n))
    assign(var.name, read.csv(conn), envir=globalenv())
    if(write.to.file){
      eval.text=paste0('write.csv(', var.name, ',"', gsub('-','_',n), '", row.names=F, fileEncoding = "UTF-8")')
      eval(parse(text=eval.text))
      }
  }
  unlink(temp)
  }

get.csv.data=function(url, write.to.file=F){
  temp=GET(url['url'])
  temp=rawToChar(temp$content)
  temp=read.csv(text=temp)
  assign(url['name'] ,temp, envir=globalenv())
  if(write.to.file){
    eval.text=paste0('write.csv(', url['name'], ',"', paste0(url['name'], '.csv'), '", row.names=F, fileEncoding = "UTF-8")')
    eval(parse(text=eval.text))
    }
  }





get.json.data('https://api.opencovid.ca/timeseries', 'ts.hr', 'loc=hr', T)
get.json.data('https://api.opencovid.ca/timeseries', 'ts.prov', 'loc=prov', T)
get.json.data('https://api.opencovid.ca/timeseries', 'ts.canada','loc=canada', write.to.file=T)

get.ind.data(case.zip, T)

get.csv.data(github.age.case.map, T)
get.csv.data(github.age.mortality.map, T)
get.csv.data(github.hr.map, T)
get.csv.data(github.prov.map, T)
