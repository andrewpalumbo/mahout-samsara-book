package nbexample


import java.io.File

import org.apache.catalina.Context
import org.apache.catalina.LifecycleException
import org.apache.catalina.startup.Tomcat

class TomcatServer {

  def startServer(port: Int): Unit =  {
    val tomcat: Tomcat = new Tomcat()

    tomcat.setPort(port)

    val base: File = new File(System.getProperty("java.io.tmpdir"))

    val rootCtx: Context  = tomcat.addContext("/app", base.getAbsolutePath())
    Tomcat.addServlet(rootCtx, "countryServlet", new NaiveBayesServlet())
    rootCtx.addServletMapping("/country", "countryServlet")
    tomcat.start()
    tomcat.getServer().await()
  }
}
