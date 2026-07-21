using FAIR.API.Interfaces.Report;
using FAIR.Application.DTOs.Report;
using FAIR.Application.Services.Implementations.Reports;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace FAIR.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize]
    public class ReportController(IReportService reportService) : ControllerBase
    {
        [HttpGet("all")]
        public async Task<ActionResult<IEnumerable<GetReport>>> GetAll()
        {
            var reports = await reportService.GetAllAsync();
            return reports.Any()? Ok(reports):NotFound();
        }
        [HttpGet("single/{id}")]
        public async Task<IActionResult> GetById(string id)
        {
            var reports = await reportService.GetByIdAsync(id);
            return reports is null ?  NotFound() : Ok(reports);
        }
        [HttpGet("player/{playerId}")]
        public async Task<ActionResult<IEnumerable<GetReport>>> GetReportsByPlayerId(string playerId)
        {
            var reports = await reportService.GetReportsByPlayerIdAsync(playerId);
            return Ok(reports);
        }
        [HttpGet("top-players/{count}")]
        public async Task<ActionResult<IEnumerable<GetPlayer>>> GetTopPlayer(int count )
        {
            if (count <= 0) return BadRequest("The count must greater than 0 ");
            var topPlayers = await reportService.GetTopPlayersByScoreAsync(count);
            return Ok(topPlayers);
        }
        [HttpPost("add")]
        public async Task<IActionResult> Add([FromBody] CreateReport report)
        {
            var resut = await reportService.AddAsync(report);
            return resut.Success ? Ok(resut) : BadRequest();
        }

        [HttpPut("update")]
        public async Task<IActionResult> Update([FromBody] UpdateReport report)
        {
            var resut = await reportService.UpdateAsync(report);
            return resut.Success ? Ok(resut) : BadRequest();
        }
        [HttpDelete("delete/{id}")]
        public async Task<IActionResult> Delete(string id)
        {
            var resut = await reportService.DeleteAsync(id);
            return resut.Success ? Ok(resut) : BadRequest();
        }

    }
}
