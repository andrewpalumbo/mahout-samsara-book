package myMahoutApp.scratch

import org.apache.log4j.{BasicConfigurator, Logger, Level}
import org.scalatest.{Matchers, FunSuite}
import sys.process._
import org.apache.mahout.logging._

/**
 * @author dmitriy
 */
class ScratchSuite extends FunSuite with Matchers {

  BasicConfigurator.configure()
  implicit val loglog = getLog(classOf[ScratchSuite])

  test ("s1") {

    setLogLevel(Level.DEBUG)

    val t = new java.util.Timer()
    val task = new java.util.TimerTask {
      def run() = None
    }
    t.schedule(task, 1000L, 1000L)

  }

}
