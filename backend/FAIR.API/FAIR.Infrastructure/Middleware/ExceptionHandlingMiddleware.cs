using FAIR.Application.Services.Interfaces.Logging;
using Microsoft.AspNetCore.Http;
using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
namespace eCommerce.Infrastructure.Middleware
{
    public class ExceptionHandlingMiddleware(RequestDelegate _next)
    {
        public async Task InvokeAsync(HttpContext context)
        {
            try
            {
                await _next(context);
            }
            catch (DbUpdateException ex)
            {
                var logger = context.RequestServices.GetRequiredService<IAppLogger<ExceptionHandlingMiddleware>>();
                context.Response.ContentType = "application/json";
                if (ex.InnerException is SqlException innerException)
                {
                    logger.LogError(innerException, "SQL Exception");
                    switch (innerException.Number)
                    {
                        case 2627:
                            context.Response.StatusCode = StatusCodes.Status409Conflict;
                            await context.Response.WriteAsync("Unique constraint violation");
                            break;
                        case 515:
                            context.Response.StatusCode = StatusCodes.Status400BadRequest;
                            await context.Response.WriteAsync("Cannot insert Null");
                            break;
                        case 547:
                            context.Response.StatusCode = StatusCodes.Status409Conflict;
                            await context.Response.WriteAsync("Foreign key constraint violation");
                            break;
                        default:
                            context.Response.StatusCode = StatusCodes.Status500InternalServerError;
                            await context.Response.WriteAsync("Error occure While Processing your request.");
                            break;
                    }
                }
                else
                {
                    logger.LogError(ex, "Related EFCore Exception");
                    context.Response.StatusCode = StatusCodes.Status500InternalServerError;
                    await context.Response.WriteAsync("Error occure While Saving the Entity Changes");

                }
            }
            catch (Exception ex)
            {
                var logger = context.RequestServices.GetRequiredService<IAppLogger<ExceptionHandlingMiddleware>>();
                logger.LogError(ex, "Unknown Exception");
                context.Response.StatusCode = StatusCodes.Status500InternalServerError;
                await context.Response.WriteAsync("Error occured: " + ex.Message);
            }

        }
    }
}
